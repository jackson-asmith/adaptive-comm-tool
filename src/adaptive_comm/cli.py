"""Command-line interface: adaptive-comm "your message here"."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import anthropic
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from adaptive_comm.analyzer import DEFAULT_EFFORT, DEFAULT_MODEL, AnalysisError, Analyzer, MessageAnalysis
from adaptive_comm.editor import edit_message
from adaptive_comm.personas import PersonaFileError, load_personas


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
        help="treat each non-empty line of --file/stdin as a separate message",
    )
    p.add_argument("-y", "--yes", action="store_true", help="send clipboard text without opening it for editing first")
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


# Clipboard readers to try, in order. The first one installed wins.
CLIPBOARD_COMMANDS = [
    ["pbpaste"],  # macOS
    ["wl-paste", "--no-newline"],  # Linux, Wayland
    ["xclip", "-selection", "clipboard", "-o"],  # Linux, X11
    ["xsel", "--clipboard", "--output"],  # Linux, X11
    ["powershell.exe", "-NoProfile", "-Command", "Get-Clipboard -Raw"],  # Windows, WSL
]


class ClipboardError(RuntimeError):
    """Raised when no clipboard tool is available or reading it fails."""


class Cancelled(Exception):
    """Raised when the user cancels out of the editor."""


def read_clipboard() -> str:
    for cmd in CLIPBOARD_COMMANDS:
        if shutil.which(cmd[0]) is None:
            continue
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise ClipboardError(f"{cmd[0]} failed: {result.stderr.strip() or f'exit code {result.returncode}'}")
        return result.stdout
    raise ClipboardError("no clipboard tool found (install wl-clipboard or xclip on Linux)")


def read_messages(args: argparse.Namespace, err: Console) -> list[str]:
    messages = list(args.messages)

    text = None
    if args.clipboard:
        text = read_clipboard()
        if not args.yes and not args.each_line and sys.stdin.isatty():
            err.print("[dim]From clipboard. Edit if needed.[/dim]")
            text = edit_message(text.strip())
            if text is None:
                raise Cancelled
    elif args.file:
        text = sys.stdin.read() if args.file == "-" else Path(args.file).read_text()
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

    try:
        messages = read_messages(args, err)
    except ClipboardError as e:
        err.print(f"[red]error:[/red] could not read the clipboard: {e}")
        return 2
    except Cancelled:
        err.print("Cancelled.")
        return 1
    if not messages:
        where = "the clipboard is empty" if args.clipboard else "pass it as an argument, pipe it in, or use --file or -c"
        err.print(f"[red]error:[/red] no message given ({where})")
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
            preview = " ".join(msg.split())
            with err.status(f"Analyzing: {preview[:60]}{'...' if len(preview) > 60 else ''}"):
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
