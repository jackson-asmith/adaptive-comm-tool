# adaptive-comm-tool

**See how your message lands with different people before you hit send.**

The same Slack message can read as efficient to one teammate and dismissive to another. `adaptive-comm` takes a message and a set of personas (the people who will read it), then uses Claude to:

- score how well the message lands with each persona (0–10)
- explain their likely reaction, in their own voice
- point out the exact words or omissions that cause friction
- **rewrite the message for each persona** while keeping your intent and your ask

## Example

```bash
adaptive-comm "Re-run the integration tests. They're failing again."
```

Illustrative output (abridged; the exact wording changes from run to run):

```
╭─ Message ─────────────────────────────────────────────╮
│ Re-run the integration tests. They're failing again.  │
╰──────────────────────────── tone: direct, terse ──────╯
┌────────────────────┬───────┬──────────────────────────────┬──────────────────────────────────────────┐
│ Persona            │ Score │ Reaction                     │ Try instead                              │
├────────────────────┼───────┼──────────────────────────────┼──────────────────────────────────────────┤
│ pragmatic_engineer │ 6/10  │ Clear ask, but which suite,  │ Can you re-run the integration suite?    │
│                    │       │ and failing how?             │ The payments tests failed twice today    │
│                    │       │ friction: no failure details │ with timeouts: <link to CI run>          │
├────────────────────┼───────┼──────────────────────────────┼──────────────────────────────────────────┤
│ empathic_pm        │ 3/10  │ Feels like an order, and     │ Hey! Could you re-run the integration    │
│                    │       │ "again" sounds like blame.   │ tests when you get a sec? They're still  │
│                    │       │ friction: "again"; no please │ flaky and I'd like to rule out a blip.   │
├────────────────────┼───────┼──────────────────────────────┼──────────────────────────────────────────┤
│ vision_director    │ 5/10  │ Fine, but is this a one-off  │ Can you re-run the integration tests?    │
│                    │       │ or a trend we should fix?    │ This is the third failure this week, so  │
│                    │       │                              │ worth a ticket to fix the flakiness.     │
└────────────────────┴───────┴──────────────────────────────┴──────────────────────────────────────────┘
```

## Install

Requires Python 3.10+ and an [Anthropic API key](https://console.anthropic.com/).

```bash
git clone https://github.com/jackson-asmith/adaptive-comm-tool.git
cd adaptive-comm-tool
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
# One or more messages as arguments
adaptive-comm "Can we push the deadline to Friday?" "LGTM 👍"

# A file with one message per line, or '-' for stdin
adaptive-comm --file drafts.txt
pbpaste | adaptive-comm --file -

# Only check against specific personas
adaptive-comm --only empathic_pm --only vision_director "Let's revert this."

# JSON output, for scripting or saving a report
adaptive-comm --json "Ship it." > report.json
adaptive-comm -o report.json "Ship it."     # tables on screen, JSON to file

# See which personas are loaded
adaptive-comm --list-personas
```

| Option | Description |
|---|---|
| `-f, --file PATH` | Read messages from a file, one per line (`-` for stdin) |
| `-p, --personas PATH` | Use your own personas YAML file |
| `--only NAME` | Only use this persona (repeatable) |
| `--json` | Print JSON instead of tables |
| `-o, --output PATH` | Also write the JSON report to a file |
| `--model` | Claude model (default `claude-opus-5-5`) |
| `--effort` | Reasoning effort: `low`, `medium` (default), `high`, `xhigh`, `max` |

## Define your own personas

The built-in personas are a pragmatic engineer, an empathic PM, and a vision-focused director (see [`personas.yaml`](src/adaptive_comm/personas.yaml)). To model your own team, write a YAML file:

```yaml
priya:
  role: Staff engineer, owns the payments service
  description: Protective of on-call load and allergic to surprise changes.
  values: [advance notice, data over opinions, respecting ownership]
  dislikes: [drive-by requests, "quick" asks that aren't quick]

sam:
  role: New grad on the team
  description: Eager, but can read a terse message as disapproval.
  values: [encouragement, explicit next steps]
  dislikes: [sarcasm, unexplained jargon]
```

```bash
adaptive-comm --personas my_team.yaml "Why was this merged without review?"
```

Each persona needs `role`, `values`, and `dislikes`. `description` is optional.

## How it works

For each message, the tool makes one request to the Claude API that includes all the personas. It uses [structured outputs](https://docs.claude.com/en/docs/build-with-claude/structured-outputs), so the response always matches a JSON schema, and persona names are pinned to an enum so Claude can't invent or drop one. The response is validated with Pydantic and rendered with [Rich](https://github.com/Textualize/rich).

```
src/adaptive_comm/
├── personas.py     # load and validate persona YAML
├── personas.yaml   # built-in personas
├── analyzer.py     # prompt, JSON schema, API call, response validation
└── cli.py          # argument parsing and table/JSON output
```

If a safety classifier declines a request, the API retries it on a fallback model instead of failing the whole run (server-side fallbacks).

## Development

```bash
pip install -e '.[dev]'
pytest
```

The tests use a fake client, so they need no API key and make no network calls.

## History

This started as a small rule-based script that matched keywords like "blunt" and "emoji" against fixed persona traits. The first commit in this repo keeps that version for comparison.

## License

MIT
