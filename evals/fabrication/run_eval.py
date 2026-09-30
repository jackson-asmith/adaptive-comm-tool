"""Fabrication eval: do rewrites add facts, promises, or claims the sender didn't make?

Runs each message through the real Analyzer, then has a judge model grade every
persona's rewrite against the original. Makes paid API calls; needs
ANTHROPIC_API_KEY (or another credential source the Anthropic SDK understands).

    python evals/fabrication/run_eval.py --check-judge        # grade known answers first
    python evals/fabrication/run_eval.py --pilot              # 5 messages
    python evals/fabrication/run_eval.py --reps 2             # the full set
    python evals/fabrication/run_eval.py --report             # rebuild the report only

Output goes to .claude/hillclimb/fabrication/<variant>/: results.jsonl (one row
per message and rep), traces/ (full exchanges, including the judge's),
errors.jsonl (attempts that failed for reasons other than the model), and
report.md. Re-running resumes: finished (message, rep) pairs are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import anthropic
import yaml

from adaptive_comm.analyzer import DEFAULT_EFFORT, DEFAULT_MODEL, EFFORT_LEVELS, SYSTEM_PROMPT, AnalysisError, Analyzer
from adaptive_comm.personas import load_personas

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FLOW_DIR = ROOT / ".claude" / "hillclimb" / "fabrication"
STATE_FILE = FLOW_DIR / "_state.json"
sys.path.insert(0, str(HERE))
from judge import JUDGE_MODEL, JUDGE_SYSTEM, JudgeError, judge, judge_prompt  # noqa: E402

# Files that define how the eval measures. A change to any of them needs
# re-approval (--approve-harness), so scores from different graders never mix.
HARNESS_PATHS = [HERE / "run_eval.py", HERE / "judge.py"]

# A small, varied slice for piloting the harness and the grader.
PILOT_IDS = ["terse-rerun-tests", "hedged-secrets", "humor-meeting-rooms", "refusal-admin-rights", "long-storage-slip"]

# $ per million tokens: (input, output). Cache writes bill at 1.25x input, reads at 0.1x.
PRICES = {"claude-opus-5-5": (4.00, 20.00), "claude-sonnet-5-5": (2.00, 10.00)}

METRICS = [
    {
        "id": "no_additions",
        "label": "No additions",
        "kind": "continuous",
        "description": "Share of the message's rewrites that add no facts, promises, or claims.",
    },
    {
        "id": "keeps_meaning",
        "label": "Keeps meaning",
        "kind": "continuous",
        "description": "Share of rewrites that keep the original's facts, ask, and stance.",
    },
    {
        "id": "additions",
        "label": "Additions",
        "kind": "count",
        "description": "Total invented items across the message's rewrites (lower is better).",
    },
]


# --- Inputs and state --------------------------------------------------------


def load_cases() -> list[dict[str, Any]]:
    cases = yaml.safe_load((HERE / "cases.yaml").read_text())
    for c in cases:
        c.setdefault("source", "variation")
    private = HERE / "cases.local.yaml"
    if private.exists():
        cases += yaml.safe_load(private.read_text()) or []
    return cases


def harness_sha() -> str:
    h = hashlib.sha256()
    for path in HARNESS_PATHS:
        h.update(path.name.encode() + b"\0" + path.read_bytes())
    return h.hexdigest()


def read_state() -> dict[str, Any]:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def write_state(state: dict[str, Any]) -> None:
    FLOW_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


def check_harness(approve: bool) -> None:
    state = read_state()
    current = harness_sha()
    if approve:
        state.update(harness_sha=current, harness_paths=[str(p.relative_to(ROOT)) for p in HARNESS_PATHS])
        state.setdefault("metrics", METRICS)
        write_state(state)
        print(f"Harness approved ({current[:12]}).")
        return
    if state.get("harness_sha") != current:
        sys.exit(
            "The runner or judge has changed since it was last approved (or was never approved).\n"
            "Review the change, then re-run with --approve-harness to accept it."
        )


def completed_keys(variant_dir: Path) -> set[tuple[str, int]]:
    path = variant_dir / "results.jsonl"
    if not path.exists():
        return set()
    return {(r["prompt_id"], r["rep"]) for r in map(json.loads, path.read_text().splitlines()) if r}


# --- Running one case --------------------------------------------------------


class HarnessFailure(RuntimeError):
    def __init__(self, failure_class: str, message: str, usage: dict[str, Any] | None = None, model: str | None = None):
        super().__init__(message)
        self.failure_class, self.usage, self.model = failure_class, usage, model


def usage_dict(response: Any) -> dict[str, int]:
    u = response.usage
    return {
        "input_tokens": u.input_tokens or 0,
        "output_tokens": u.output_tokens or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }


def add_usage(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


def served_model_ok(requested: str, served: str) -> bool:
    return served == requested or served.startswith(requested + "-")


def run_case(
    case: dict[str, Any], rep: int, args: argparse.Namespace, client: anthropic.Anthropic, variant_dir: Path
) -> dict[str, Any]:
    personas = load_personas()
    analyzer = Analyzer(personas, client=client, model=args.model, effort=args.effort)
    message = case["message"]

    start = time.monotonic()
    try:
        analysis = analyzer.analyze(message)
        failure = None
    except AnalysisError as e:
        analysis, failure = None, e
    except anthropic.APIError as e:
        raise HarnessFailure("api_error", str(e)) from e
    latency = time.monotonic() - start

    response = analyzer.last_response
    if response is None:
        raise HarnessFailure("api_error", "no response recorded")
    usage = usage_dict(response)
    if not served_model_ok(args.model, response.model):
        raise HarnessFailure(
            "served_model_mismatch", f"asked for {args.model}, served by {response.model}", usage, response.model
        )

    trace: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
        {"role": "assistant", "content": "".join(b.text for b in response.content if b.type == "text")},
    ]
    row: dict[str, Any] = {
        "prompt_id": case["id"],
        "rep": rep,
        "prompt": message,
        "tags": [case["category"], case["source"]],
        "model": response.model,
        "usage": usage,
        "stop_reason": response.stop_reason,
        "latency_s": round(latency, 2),
        # Which runner and grader produced this row; grades from different versions aren't comparable.
        "harness_sha": harness_sha()[:12],
    }

    if analysis is None:
        # A refusal or a cut-off answer is the model's outcome, not a harness failure:
        # it's recorded and shown, but left out of the score averages.
        row["status"] = (
            "truncated"
            if response.stop_reason == "max_tokens"
            else ("refused" if response.stop_reason == "refusal" else "unparseable")
        )
        row["grade"] = {}
        row["meta"] = {"error": str(failure)}
        write_trace(variant_dir, case["id"], rep, trace)
        return row

    rewrites = [(r.persona, r.rewrite) for r in analysis.reactions]
    row.update(grade_rewrites(client, message, rewrites, args.judge_model, trace, usage, response.model))
    write_trace(variant_dir, case["id"], rep, trace)
    return row


def grade_rewrites(
    client: anthropic.Anthropic,
    message: str,
    rewrites: list[tuple[str, str]],
    judge_model: str,
    trace: list[dict[str, Any]],
    app_usage: dict[str, int] | None = None,
    app_model: str | None = None,
) -> dict[str, Any]:
    """Judge each (persona, rewrite); append the exchanges to trace; return the row's grading fields."""
    verdicts: list[dict[str, Any]] = []
    judge_usage: dict[str, int] = {}
    trace.append({"role": "system", "name": "judge instructions", "content": JUDGE_SYSTEM})
    for persona, rewrite in rewrites:
        try:
            verdict, judge_response = judge(client, message, rewrite, model=judge_model)
        except (JudgeError, anthropic.APIError) as e:
            raise HarnessFailure("grader_error", f"judging {persona}: {e}", app_usage, app_model) from e
        if not served_model_ok(judge_model, judge_response.model):
            raise HarnessFailure(
                "served_model_mismatch", f"judge served by {judge_response.model}", app_usage, app_model
            )
        judge_usage = add_usage(judge_usage, usage_dict(judge_response))
        verdicts.append({"persona": persona, "rewrite": rewrite, **verdict})
        trace += [
            {"role": "user", "name": f"judge: {persona}", "content": judge_prompt(message, rewrite)},
            {"role": "assistant", "name": "judge verdict", "content": json.dumps(verdict, indent=2)},
        ]

    n = len(verdicts)
    added = "; ".join(f'{v["persona"]}: {a["kind"]} "{a["quote"]}"' for v in verdicts for a in v["additions"])
    return {
        "status": "ok",
        "judge_model": judge_model,
        "judge_usage": judge_usage,
        "grade": {
            "no_additions": sum(not v["additions"] for v in verdicts) / n,
            "keeps_meaning": sum(v["keeps_meaning"] for v in verdicts) / n,
            "additions": sum(len(v["additions"]) for v in verdicts),
        },
        "explanation": {"no_additions": added or "none"},
        "meta": {"verdicts": verdicts},
    }


def regrade(args: argparse.Namespace) -> None:
    """Re-judge the rewrites from an earlier run with the current grader, without new app calls."""
    source = Path(args.regrade_from)
    old_rows = [json.loads(line) for line in (source / "results.jsonl").read_text().splitlines() if line.strip()]
    variant_dir = FLOW_DIR / args.variant
    if (variant_dir / "results.jsonl").exists():
        sys.exit(f"{variant_dir} already has results; move them aside or choose another --variant.")
    variant_dir.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic(max_retries=4, timeout=args.timeout_s)
    recorder = Recorder(variant_dir, sum(r["status"] == "ok" for r in old_rows))
    app_fields = ("prompt_id", "rep", "prompt", "tags", "model", "usage", "stop_reason", "latency_s")
    for old in old_rows:
        if old["status"] != "ok":
            continue
        # Keep the app's side of the old transcript, then add fresh judge exchanges.
        old_trace = json.loads((source / "traces" / f"{old['prompt_id']}_rep{old['rep']}.json").read_text())
        trace = old_trace[:3]
        rewrites = [(v["persona"], v["rewrite"]) for v in old["meta"]["verdicts"]]
        row = {k: old[k] for k in app_fields} | {"harness_sha": harness_sha()[:12]}
        try:
            row |= grade_rewrites(client, old["prompt"], rewrites, args.judge_model, trace, old["usage"], old["model"])
        except HarnessFailure as e:
            recorder.error(old["prompt_id"], old["rep"], e)
            continue
        row["meta"]["regraded_from"] = str(source)
        write_trace(variant_dir, old["prompt_id"], old["rep"], trace)
        recorder.ok(row)
    write_report(args.variant)


def write_trace(variant_dir: Path, case_id: str, rep: int, trace: list[dict[str, Any]]) -> None:
    traces = variant_dir / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    (traces / f"{case_id}_rep{rep}.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n")


# --- Running the set ---------------------------------------------------------


class Recorder:
    """Writes each finished (message, rep) as it completes, so a crash loses nothing."""

    def __init__(self, variant_dir: Path, total: int) -> None:
        self.results, self.errors = variant_dir / "results.jsonl", variant_dir / "errors.jsonl"
        self.total, self.count = total, 0
        self.lock = threading.Lock()

    def _append(self, path: Path, row: dict[str, Any]) -> None:
        with self.lock, path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def ok(self, row: dict[str, Any]) -> None:
        self._append(self.results, row)
        self.count += 1
        score = row["grade"].get("no_additions")
        shown = f"{score:.2f}" if score is not None else row["status"]
        print(f"  [{self.count}/{self.total}] {row['prompt_id']} rep {row['rep']}: {shown}")

    def error(self, case_id: str, rep: int, failure: HarnessFailure) -> None:
        # Failed attempts never go in results.jsonl, so a re-run retries them.
        self._append(
            self.errors,
            {
                "prompt_id": case_id,
                "rep": rep,
                "failure_class": failure.failure_class,
                "error": str(failure),
                "model": failure.model,
                "usage": failure.usage,
            },
        )
        self.count += 1
        print(f"  [{self.count}/{self.total}] {case_id} rep {rep}: ERROR {failure.failure_class}: {failure}")


def select_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases = load_cases()
    if args.pilot:
        # Some pilot messages are private; top up from the rest if they're absent.
        chosen = [c for c in cases if c["id"] in PILOT_IDS]
        chosen += [c for c in cases if c not in chosen][: len(PILOT_IDS) - len(chosen)]
        cases = chosen
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]
    if not cases:
        sys.exit("No cases selected.")
    return cases


def run(args: argparse.Namespace) -> None:
    cases = select_cases(args)
    variant_dir = FLOW_DIR / args.variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    done = completed_keys(variant_dir)
    todo = [(c, rep) for c in cases for rep in range(args.reps) if (c["id"], rep) not in done]
    skipped = len(cases) * args.reps - len(todo)
    print(f"{len(cases)} messages x {args.reps} rep(s): {len(todo)} to run, {skipped} already done.")

    # The SDK retries 429/5xx with jittered backoff; the timeout keeps one request from hanging a worker.
    client = anthropic.Anthropic(max_retries=4, timeout=args.timeout_s)
    recorder = Recorder(variant_dir, len(todo))
    wall_start = time.monotonic()

    pool = ThreadPoolExecutor(max_workers=args.concurrency)
    labels: dict[Future[dict[str, Any]], tuple[str, int]] = {
        pool.submit(run_case, case, rep, args, client, variant_dir): (case["id"], rep) for case, rep in todo
    }
    started: dict[Future[dict[str, Any]], float] = {}
    pending = set(labels)
    try:
        while pending:
            time.sleep(0.5)
            for fut in list(pending):
                if fut.running():
                    started.setdefault(fut, time.monotonic())
                case_id, rep = labels[fut]
                if fut.done():
                    pending.discard(fut)
                    try:
                        recorder.ok(fut.result())
                    except HarnessFailure as e:
                        recorder.error(case_id, rep, e)
                elif fut in started and time.monotonic() - started[fut] > args.timeout_s:
                    pending.discard(fut)  # the thread can't be killed; its eventual result is ignored
                    recorder.error(case_id, rep, HarnessFailure("timeout", f"exceeded {args.timeout_s:.0f}s"))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    print(f"Wall clock: {time.monotonic() - wall_start:.0f}s")
    write_report(args.variant)


# --- Summary and report ------------------------------------------------------


def row_cost(row: dict[str, Any]) -> float:
    total = 0.0
    for model_key, usage_key in (("model", "usage"), ("judge_model", "judge_usage")):
        model, usage = row.get(model_key), row.get(usage_key) or {}
        if not model:
            continue
        base = next((p for m, p in PRICES.items() if model.startswith(m)), None)
        if base is None:
            raise SystemExit(f"No price for {model}; add it to PRICES.")
        price_in, price_out = base
        total += (
            usage.get("input_tokens", 0) * price_in
            + usage.get("output_tokens", 0) * price_out
            + usage.get("cache_creation_input_tokens", 0) * price_in * 1.25
            + usage.get("cache_read_input_tokens", 0) * price_in * 0.1
        ) / 1e6
    return total


def mean_ci(values: list[float]) -> tuple[float, float]:
    """Mean and 95% half-width (normal approximation over messages)."""
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], float("nan")
    return statistics.fmean(values), 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def write_report(variant: str) -> None:
    variant_dir = FLOW_DIR / variant
    rows = (
        [json.loads(line) for line in (variant_dir / "results.jsonl").read_text().splitlines() if line.strip()]
        if (variant_dir / "results.jsonl").exists()
        else []
    )
    errors = (
        [json.loads(line) for line in (variant_dir / "errors.jsonl").read_text().splitlines() if line.strip()]
        if (variant_dir / "errors.jsonl").exists()
        else []
    )
    versions = {r.get("harness_sha") for r in rows}
    if len(versions) > 1:
        raise SystemExit(
            f"{variant} mixes results from {len(versions)} runner/grader versions; "
            "move the old results aside or use a new --variant."
        )
    ok = [r for r in rows if r["status"] == "ok"]

    # Fail loudly on rows that claim success but are missing what the report needs.
    for r in ok:
        missing = [k for k in ("model", "usage", "judge_model", "judge_usage", "grade") if not r.get(k)]
        if missing or not r["usage"].get("output_tokens"):
            raise SystemExit(f"Row {r['prompt_id']} rep {r['rep']} is missing {missing or ['usage']}: runner bug.")

    by_case: dict[str, list[dict[str, Any]]] = {}
    for r in ok:
        by_case.setdefault(r["prompt_id"], []).append(r)

    def case_mean(metric: str) -> list[float]:
        return [statistics.fmean(r["grade"][metric] for r in rs) for rs in by_case.values()]

    no_add, no_add_ci = mean_ci(case_mean("no_additions"))
    keeps, keeps_ci = mean_ci(case_mean("keeps_meaning"))
    costs = [row_cost(r) for r in rows]
    latencies = sorted(r["latency_s"] for r in rows)
    other = {s: sum(r["status"] == s for r in rows) for s in ("refused", "truncated", "unparseable")}

    lines = [
        f"# Fabrication eval: {variant}",
        "",
        f"**No additions: {no_add:.0%} ± {no_add_ci:.0%}** of rewrites add nothing the sender didn't say "
        f"({len(by_case)} messages, {len(ok)} graded runs, 95% interval over messages).",
        "",
        f"- Keeps meaning: {keeps:.0%} ± {keeps_ci:.0%}",
        f"- Total invented items: {sum(r['grade']['additions'] for r in ok)}",
        f"- Not graded: {other['refused']} refused, {other['truncated']} cut off, {other['unparseable']} unparseable; "
        f"{len(errors)} harness errors (see errors.jsonl)",
        f"- Cost: ${sum(costs):.2f} total, ${statistics.fmean(costs):.3f} per run (app + judge)"
        if costs
        else "- Cost: n/a",
        f"- App latency: median {statistics.median(latencies):.1f}s, max {latencies[-1]:.1f}s" if latencies else "",
        "",
        "| Message | Category | Source | No additions | Keeps meaning | What was added | Trace |",
        "|---|---|---|---|---|---|---|",
    ]

    def fmt(value: Any) -> str:
        return f"{value:.2f}" if isinstance(value, float) else "-"

    for r in sorted(rows, key=lambda r: (r["tags"][0], r["prompt_id"], r["rep"])):
        g = r["grade"]
        added = r.get("explanation", {}).get("no_additions", r["status"]).replace("|", "\\|")
        trace = f"{variant}/traces/{r['prompt_id']}_rep{r['rep']}.json"
        lines.append(
            f"| {r['prompt_id']} (rep {r['rep']}) | {r['tags'][0]} | {r['tags'][1]} | "
            f"{fmt(g.get('no_additions'))} | {fmt(g.get('keeps_meaning'))} | {added} | [trace](./{trace}) |"
        )
    report = FLOW_DIR / "report.md"
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:9]))
    print(f"\nReport: {report}")


# --- Judge check -------------------------------------------------------------


def check_judge(args: argparse.Namespace) -> None:
    """Grade the known answers, args.judge_reps times each, and report agreement and consistency."""
    checks = yaml.safe_load((HERE / "judge_check.yaml").read_text())
    client = anthropic.Anthropic(max_retries=4)
    agree = total = 0
    unstable = []
    usage: dict[str, int] = {}
    for i, check in enumerate(checks, 1):
        seen: list[dict[str, bool]] = []
        for _ in range(args.judge_reps):
            verdict, response = judge(client, check["original"], check["rewrite"], model=args.judge_model)
            usage = add_usage(usage, usage_dict(response))
            got = {"additions": bool(verdict["additions"]), "keeps_meaning": verdict["keeps_meaning"]}
            seen.append(got)
            results = []
            for key, expected in check["expect"].items():
                if expected is None:
                    continue
                total += 1
                agree += got[key] == expected
                results.append(f"{key} {'ok' if got[key] == expected else f'WRONG (expected {expected})'}")
            found = "; ".join(f'{a["kind"]}: "{a["quote"]}"' for a in verdict["additions"]) or "no additions"
            print(f"{i:2}. {check['note']}: {', '.join(results)}\n    judge found: {found}")
        if any(s != seen[0] for s in seen):
            unstable.append(f"{i}. {check['note']}")
    cost = row_cost({"judge_model": args.judge_model, "judge_usage": usage})
    print(f"\nJudge agreed on {agree}/{total} checks ({agree / total:.0%}). Cost ${cost:.3f}.")
    if args.judge_reps > 1:
        print(
            f"Consistency: {len(checks) - len(unstable)}/{len(checks)} known answers got the same verdict "
            f"on all {args.judge_reps} runs."
        )
        for item in unstable:
            print(f"  varied: {item}")


# --- CLI ---------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="baseline", help="output directory name: baseline, v1, v2, ...")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--effort", default=DEFAULT_EFFORT, choices=EFFORT_LEVELS)
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--pilot", action="store_true", help=f"only the {len(PILOT_IDS)} pilot messages")
    p.add_argument("--only", action="append", metavar="ID", help="only this message (repeatable)")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout-s", type=float, default=300, help="wall-clock ceiling per message")
    p.add_argument("--check-judge", action="store_true", help="grade the known answers in judge_check.yaml")
    p.add_argument("--judge-reps", type=int, default=1, help="with --check-judge, grade each known answer N times")
    p.add_argument("--report", action="store_true", help="rebuild the report from existing results")
    p.add_argument("--regrade-from", metavar="DIR", help="re-judge the rewrites in DIR/results.jsonl (no app calls)")
    p.add_argument("--approve-harness", action="store_true", help="accept the current runner and judge, then exit")
    args = p.parse_args()

    if args.report:
        write_report(args.variant)
        return
    check_harness(args.approve_harness)
    if args.approve_harness:
        return
    if args.check_judge:
        check_judge(args)
    elif args.regrade_from:
        regrade(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
