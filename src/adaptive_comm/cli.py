"""Command-line interface: adaptive-comm "your message here"."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import anthropic
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from adaptive_comm import __version__
from adaptive_comm.analyzer import (
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    AnalysisError,
    Analyzer,
    MessageAnalysis,
    MissingCredentialsError,
)
from adaptive_comm.clipboard import ClipboardError, read_clipboard, write_clipboard
from adaptive_comm.editor import edit_message
from adaptive_comm.personas import Persona, PersonaFileError, load_personas


class CLIError(Exception):
    """A problem to report to the user as `error: <message>`, exiting with `code`."""

    def __init__(self, message: str, code: int = 2):
        super().__init__(message)
        self.code = code


class InputError(CLIError):
    """Raised when the message can't be read (missing file, not text, clipboard failure)."""


class Cancelled(Exception):
    """Raised when the user cancels out of the editor."""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="adaptive-comm",
        description="See how a message lands with different personas, and get a rewrite for each.",
    )
    p.add_argument(
        "messages",
        nargs="*",
        help="message(s) to analyze; if omitted, reads piped stdin or opens an editor to paste into",
    )
    source = p.add_mutually_exclusive_group()
    source.add_argument("-f", "--file", help="read a message from a file ('-' for stdin)")
    source.add_argument("-c", "--clipboard", action="store_true", help="read the message from the clipboard")
    p.add_argument(
        "--each-line",
        action="store_true",
        help="treat each non-empty line of --file, -c, or piped input as a separate message",
    )
    p.add_argument("-y", "--yes", action="store_true", help="with -c, send the clipboard text without editing it first")
    p.add_argument("-p", "--personas", help="YAML file of personas (default: built-in set)")
    p.add_argument(
        "--only",
        action="append",
        metavar="NAME",
        help="only use this persona (repeatable)",
    )
    p.add_argument("--list-personas", action="store_true", help="show the loaded personas and exit")
    p.add_argument("--json", action="store_true", help="print the report as JSON instead of tables")
    p.add_argument("-o", "--output", help="also write the JSON report to this file")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model (default: {DEFAULT_MODEL})")
    p.add_argument(
        "--effort",
        default=DEFAULT_EFFORT,
        choices=["low", "medium", "high", "xhigh", "max"],
        help=f"reasoning effort (default: {DEFAULT_EFFORT})",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def check_flags(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject flag combinations that would otherwise be silently ignored."""
    if args.yes and not args.clipboard:
        parser.error("-y/--yes only applies with -c/--clipboard")
    if args.each_line and args.messages and not (args.file or args.clipboard):
        parser.error("--each-line applies to --file, -c, or piped input, not to messages given as arguments")


def output_path_problem(path: str) -> str | None:
    """Why the report can't be written to path, or None if it looks writable."""
    p = Path(path)
    if p.is_dir():
        return f"{path} is a directory"
    parent = p.parent
    if not parent.is_dir():
        return f"directory {parent} does not exist"
    if p.exists() and not os.access(p, os.W_OK):
        return f"{path} is not writable"
    if not p.exists() and not os.access(parent, os.W_OK):
        return f"directory {parent} is not writable"
    return None


def read_file(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    try:
        return Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise InputError(f"{path} is not a UTF-8 text file") from None
    except OSError as e:
        raise InputError(f"cannot read {path}: {e.strerror or e}") from None


def edit_or_cancel(initial: str = "") -> str:
    text = edit_message(initial)
    if text is None:
        raise Cancelled
    return text


def read_input_text(args: argparse.Namespace, err: Console) -> str | None:
    """Text from the clipboard, a file, the editor, or a pipe; None if messages came as arguments."""
    if args.clipboard:
        try:
            text = read_clipboard()
        except ClipboardError as e:
            raise InputError(f"could not read the clipboard: {e}") from None
        if args.yes or args.each_line or not sys.stdin.isatty():
            return text
        err.print("[dim]From clipboard. Edit if needed.[/dim]")
        return edit_or_cancel(text.strip())
    if args.file:
        return read_file(args.file)
    if args.messages:
        return None
    if sys.stdin.isatty():
        err.print("[dim]Paste or type your message below.[/dim]")
        return edit_or_cancel()
    return sys.stdin.read()


def read_messages(args: argparse.Namespace, err: Console) -> list[str]:
    messages = list(args.messages)
    text = read_input_text(args, err)
    if text is not None:
        if args.each_line:
            messages += [line.strip() for line in text.splitlines() if line.strip()]
        elif text.strip():
            messages.append(text.strip())
    return messages


def score_style(score: int) -> str:
    if score >= 8:
        return "bold green"
    if score >= 5:
        return "yellow"
    return "bold red"


def render(console: Console, analysis: MessageAnalysis, first_number: int = 1) -> None:
    tags = escape(", ".join(analysis.tone_tags) or "none")
    console.print(Panel(escape(analysis.message), title="Message", expand=False))
    console.print(f"[dim]tone:[/dim] {tags}")

    table = Table(show_lines=True, expand=True)
    table.add_column("#", justify="right", style="bold")
    table.add_column("Persona", style="cyan", no_wrap=True)
    table.add_column("Score", justify="center")
    table.add_column("Reaction", ratio=2)
    table.add_column("Try instead", ratio=3, style="green")
    for number, r in enumerate(analysis.reactions, start=first_number):
        reaction = escape(r.reaction)
        if r.friction_points:
            reaction += "\n[dim]friction: " + escape("; ".join(r.friction_points)) + "[/dim]"
        table.add_row(
            str(number),
            r.persona,
            f"[{score_style(r.rapport_score)}]{r.rapport_score}/10[/]",
            reaction,
            escape(r.rewrite),
        )
    console.print(table)
    console.print()


def interactive() -> bool:
    """True when a person is at the terminal on both ends (not piped or redirected)."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def offer_copy(err: Console, results: list[MessageAnalysis]) -> None:
    """Let the user copy individual rewrites to the clipboard by number."""
    choices = []
    for i, analysis in enumerate(results, start=1):
        for r in analysis.reactions:
            label = r.persona if len(results) == 1 else f"message {i}, {r.persona}"
            choices.append((label, r.rewrite))

    prompt = f"Copy a rewrite? Enter 1-{len(choices)}, or press Enter to finish: "
    while True:
        try:
            answer = err.input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            err.print()
            return
        if not answer:
            return
        if not answer.isdigit() or not 1 <= int(answer) <= len(choices):
            err.print(f"[yellow]Enter a number from 1 to {len(choices)}.[/yellow]")
            continue
        label, rewrite = choices[int(answer) - 1]
        try:
            write_clipboard(rewrite)
        except ClipboardError as e:
            err.print(f"[red]error:[/red] could not copy: {e}")
            return
        err.print(f"[green]Copied {label}'s rewrite.[/green]")


def select_personas(args: argparse.Namespace) -> list[Persona]:
    try:
        personas = load_personas(args.personas)
    except PersonaFileError as e:
        raise CLIError(str(e)) from None
    if args.only:
        unknown = sorted(set(args.only) - {p.name for p in personas})
        if unknown:
            raise CLIError(f"unknown persona(s): {', '.join(unknown)}")
        personas = [p for p in personas if p.name in args.only]
    return personas


def print_personas(console: Console, personas: list[Persona]) -> None:
    for p in personas:
        console.print(f"[cyan]{p.name}[/cyan] - {p.role}")
        console.print(f"  values:   {', '.join(p.values)}")
        console.print(f"  dislikes: {', '.join(p.dislikes)}")


def prepare_analyzer(args: argparse.Namespace, personas: list[Persona], client: anthropic.Anthropic | None) -> Analyzer:
    """Check everything that could fail before the user types a message or pays for a call."""
    if args.output and (problem := output_path_problem(args.output)):
        raise CLIError(f"can't write report: {problem}")
    try:
        return Analyzer(personas, client=client, model=args.model, effort=args.effort)
    except MissingCredentialsError as e:
        detail = "" if str(e) == "no Anthropic credentials found" else f"\n  ({e})"
        raise CLIError(f"no Anthropic credentials found. Set ANTHROPIC_API_KEY and try again.{detail}") from None


def get_messages(args: argparse.Namespace, err: Console) -> list[str]:
    messages = read_messages(args, err)
    if not messages:
        if args.clipboard:
            raise CLIError("no message given (the clipboard is empty)")
        raise CLIError("no message given (pass it as an argument, pipe it in, or use --file or -c)")
    return messages


@dataclass
class RunOutcome:
    total: int
    results: list[MessageAnalysis]
    failures: int = 0
    stopped: str | None = None  # why the run ended early, if it did


def run_analyses(
    analyzer: Analyzer, messages: list[str], console: Console, err: Console, show_tables: bool
) -> RunOutcome:
    """Analyze each message in turn. A failure that ends the run keeps what already finished."""
    outcome = RunOutcome(total=len(messages), results=[])
    for msg in messages:
        preview = " ".join(msg.split())
        try:
            with err.status(f"Analyzing: {preview[:60]}{'...' if len(preview) > 60 else ''}"):
                analysis = analyzer.analyze(msg)
        except AnalysisError as e:
            err.print(f"[red]skipped:[/red] {preview[:60]!r}: {e}")
            outcome.failures += 1
            continue
        except anthropic.AuthenticationError:
            outcome.stopped = "authentication failed. Check ANTHROPIC_API_KEY and try again."
            break
        except anthropic.APIError as e:
            outcome.stopped = f"API request failed: {e}"
            break
        except KeyboardInterrupt:
            outcome.stopped = "interrupted."
            break

        if show_tables:
            render(console, analysis, first_number=sum(len(r.reactions) for r in outcome.results) + 1)
        outcome.results.append(analysis)
    return outcome


def finish(args: argparse.Namespace, outcome: RunOutcome, err: Console) -> int:
    """Report how the run ended, offer copying, and output the JSON report. Returns the exit code."""
    results = outcome.results
    if outcome.stopped:
        err.print(f"[red]error:[/red] {outcome.stopped}")
        if outcome.total > 1:
            done = len(results) + outcome.failures
            err.print(f"Stopped after {done} of {outcome.total} messages; completed results are kept.")

    if results and not args.json and not outcome.stopped and interactive():
        offer_copy(err, results)

    report = json.dumps([r.model_dump() for r in results], indent=2)
    if args.json:
        print(report)
    if args.output and results:
        try:
            Path(args.output).write_text(report + "\n", encoding="utf-8")
        except OSError as e:
            raise CLIError(f"can't write report to {args.output}: {e.strerror or e}", code=1) from None
        err.print(f"Report written to {Path(args.output).resolve()}")

    return 1 if outcome.failures or outcome.stopped else 0


def main(argv: list[str] | None = None, client: anthropic.Anthropic | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    check_flags(parser, args)
    # soft_wrap: let the terminal wrap messages instead of Rich inserting line breaks,
    # which would split error text in logs and pipes.
    console, err = Console(), Console(stderr=True, soft_wrap=True)

    try:
        personas = select_personas(args)
        if args.list_personas:
            print_personas(console, personas)
            return 0
        analyzer = prepare_analyzer(args, personas, client)
        messages = get_messages(args, err)
        outcome = run_analyses(analyzer, messages, console, err, show_tables=not args.json)
        return finish(args, outcome, err)
    except CLIError as e:
        err.print(f"[red]error:[/red] {e}")
        return e.code
    except Cancelled:
        err.print("Cancelled.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
