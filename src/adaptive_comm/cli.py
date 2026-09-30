"""Command-line interface: adaptive-comm "your message here"."""

from __future__ import annotations

import argparse
import json
import os
import sys
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
from adaptive_comm.personas import PersonaFileError, load_personas


class InputError(RuntimeError):
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


def read_messages(args: argparse.Namespace, err: Console) -> list[str]:
    messages = list(args.messages)

    text = None
    if args.clipboard:
        try:
            text = read_clipboard()
        except ClipboardError as e:
            raise InputError(f"could not read the clipboard: {e}") from None
        if not args.yes and not args.each_line and sys.stdin.isatty():
            err.print("[dim]From clipboard. Edit if needed.[/dim]")
            text = edit_message(text.strip())
            if text is None:
                raise Cancelled
    elif args.file:
        text = read_file(args.file)
    elif not messages:
        if sys.stdin.isatty():
            err.print("[dim]Paste or type your message below.[/dim]")
            text = edit_message()
            if text is None:
                raise Cancelled
        else:
            text = sys.stdin.read()

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


def main(argv: list[str] | None = None, client: anthropic.Anthropic | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    check_flags(parser, args)
    console = Console()
    err = Console(stderr=True)

    def fail(message: str, code: int = 2) -> int:
        err.print(f"[red]error:[/red] {message}")
        return code

    try:
        personas = load_personas(args.personas)
    except PersonaFileError as e:
        return fail(str(e))

    if args.only:
        unknown = sorted(set(args.only) - {p.name for p in personas})
        if unknown:
            return fail(f"unknown persona(s): {', '.join(unknown)}")
        personas = [p for p in personas if p.name in args.only]

    if args.list_personas:
        for p in personas:
            console.print(f"[cyan]{p.name}[/cyan] - {p.role}")
            console.print(f"  values:   {', '.join(p.values)}")
            console.print(f"  dislikes: {', '.join(p.dislikes)}")
        return 0

    # Check everything that could fail before the user types a message or pays for a call.
    if args.output and (problem := output_path_problem(args.output)):
        return fail(f"can't write report: {problem}")
    try:
        analyzer = Analyzer(personas, client=client, model=args.model, effort=args.effort)
    except MissingCredentialsError as e:
        detail = "" if str(e) == "no Anthropic credentials found" else f"\n  ({e})"
        return fail(f"no Anthropic credentials found. Set ANTHROPIC_API_KEY and try again.{detail}")

    try:
        messages = read_messages(args, err)
    except InputError as e:
        return fail(str(e))
    except Cancelled:
        err.print("Cancelled.")
        return 1
    if not messages:
        where = "the clipboard is empty" if args.clipboard else "pass it as an argument, pipe it in, or use --file or -c"
        return fail(f"no message given ({where})")

    results: list[MessageAnalysis] = []
    failures = 0
    stopped = None
    for msg in messages:
        preview = " ".join(msg.split())
        try:
            with err.status(f"Analyzing: {preview[:60]}{'...' if len(preview) > 60 else ''}"):
                analysis = analyzer.analyze(msg)
        except AnalysisError as e:
            err.print(f"[red]skipped:[/red] {preview[:60]!r}: {e}")
            failures += 1
            continue
        except anthropic.AuthenticationError:
            stopped = "authentication failed. Check ANTHROPIC_API_KEY and try again."
            break
        except anthropic.APIError as e:
            stopped = f"API request failed: {e}"
            break
        except KeyboardInterrupt:
            stopped = "interrupted."
            break

        if not args.json:
            render(console, analysis, first_number=sum(len(r.reactions) for r in results) + 1)
        results.append(analysis)

    # Anything that finished is still reported, even if a later message stopped the run.
    if stopped:
        err.print(f"[red]error:[/red] {stopped}")
        if len(messages) > 1:
            done = len(results) + failures
            err.print(f"Stopped after {done} of {len(messages)} messages; completed results are kept.")

    if results and not args.json and not stopped and interactive():
        offer_copy(err, results)

    report = json.dumps([r.model_dump() for r in results], indent=2)
    if args.json:
        print(report)
    if args.output and results:
        try:
            Path(args.output).write_text(report + "\n", encoding="utf-8")
        except OSError as e:
            return fail(f"can't write report to {args.output}: {e.strerror or e}", code=1)
        err.print(f"Report written to {Path(args.output).resolve()}")

    return 1 if failures or stopped else 0


if __name__ == "__main__":
    sys.exit(main())
