"""Command-line interface: adaptive-comm "your message here"."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anthropic
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from adaptive_comm.analyzer import DEFAULT_EFFORT, DEFAULT_MODEL, AnalysisError, Analyzer, MessageAnalysis
from adaptive_comm.personas import PersonaFileError, load_personas


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="adaptive-comm",
        description="See how a message lands with different personas, and get a rewrite for each.",
    )
    p.add_argument("messages", nargs="*", help="message(s) to analyze")
    p.add_argument("-f", "--file", help="read messages from a file, one per line ('-' for stdin)")
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
    return p


def read_messages(args: argparse.Namespace) -> list[str]:
    messages = list(args.messages)
    if args.file:
        text = sys.stdin.read() if args.file == "-" else Path(args.file).read_text()
        messages += [line.strip() for line in text.splitlines() if line.strip()]
    return messages


def score_style(score: int) -> str:
    if score >= 8:
        return "bold green"
    if score >= 5:
        return "yellow"
    return "bold red"


def render(console: Console, analysis: MessageAnalysis) -> None:
    tags = ", ".join(analysis.tone_tags) or "none"
    console.print(Panel(analysis.message, title="Message", subtitle=f"tone: {tags}", expand=False))

    table = Table(show_lines=True, expand=True)
    table.add_column("Persona", style="cyan", no_wrap=True)
    table.add_column("Score", justify="center")
    table.add_column("Reaction", ratio=2)
    table.add_column("Try instead", ratio=3, style="green")
    for r in analysis.reactions:
        reaction = r.reaction
        if r.friction_points:
            reaction += "\n[dim]friction: " + "; ".join(r.friction_points) + "[/dim]"
        table.add_row(r.persona, f"[{score_style(r.rapport_score)}]{r.rapport_score}/10[/]", reaction, r.rewrite)
    console.print(table)
    console.print()


def main(argv: list[str] | None = None, client: anthropic.Anthropic | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    err = Console(stderr=True)

    try:
        personas = load_personas(args.personas)
    except PersonaFileError as e:
        err.print(f"[red]error:[/red] {e}")
        return 2

    if args.only:
        unknown = sorted(set(args.only) - {p.name for p in personas})
        if unknown:
            err.print(f"[red]error:[/red] unknown persona(s): {', '.join(unknown)}")
            return 2
        personas = [p for p in personas if p.name in args.only]

    if args.list_personas:
        for p in personas:
            console.print(f"[cyan]{p.name}[/cyan] - {p.role}")
            console.print(f"  values:   {', '.join(p.values)}")
            console.print(f"  dislikes: {', '.join(p.dislikes)}")
        return 0

    messages = read_messages(args)
    if not messages:
        err.print("[red]error:[/red] no messages given (pass them as arguments or use --file)")
        return 2

    try:
        analyzer = Analyzer(personas, client=client, model=args.model, effort=args.effort)
    except anthropic.AnthropicError as e:
        err.print(f"[red]error:[/red] could not create Anthropic client: {e}")
        return 2

    results: list[MessageAnalysis] = []
    failures = 0
    for msg in messages:
        try:
            with err.status(f"Analyzing: {msg[:60]}{'...' if len(msg) > 60 else ''}"):
                analysis = analyzer.analyze(msg)
        except AnalysisError as e:
            err.print(f"[red]skipped:[/red] {msg!r}: {e}")
            failures += 1
            continue
        except anthropic.AuthenticationError:
            err.print("[red]error:[/red] authentication failed. Check ANTHROPIC_API_KEY and try again.")
            return 2
        except TypeError as e:
            # The SDK raises TypeError when it finds no credentials at all.
            if "authentication method" not in str(e):
                raise
            err.print("[red]error:[/red] no Anthropic credentials found. Set ANTHROPIC_API_KEY and try again.")
            return 2
        except anthropic.APIError as e:
            err.print(f"[red]error:[/red] API request failed: {e}")
            return 1

        results.append(analysis)
        if not args.json:
            render(console, analysis)

    report = [r.model_dump() for r in results]
    if args.json:
        print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        err.print(f"Report written to {Path(args.output).resolve()}")

    return 1 if failures and not results else 0


if __name__ == "__main__":
    sys.exit(main())
