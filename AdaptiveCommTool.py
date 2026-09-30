import json
from pathlib import Path

# Simulate stored messages or summaries
SAMPLE_MESSAGES = [
    "Asked teammate to re-run test without giving context.",
    "Pushed back on a proposed change with strong language.",
    "Clarified design intent bluntly during Slack huddle.",
    "Requested deadline adjustment via terse one-liner.",
    "Agreed with a suggestion and added emoji for support."
]

# Personas to simulate interpretation
PERSONAS = {
    "pragmatic_engineer": {
        "values": ["clarity", "efficiency"],
        "dislikes": ["ambiguity", "unnecessary emotion"]
    },
    "empathic_pm": {
        "values": ["collaboration", "positive tone"],
        "dislikes": ["bluntness", "lack of emotional framing"]
    },
    "vision_director": {
        "values": ["strategic framing", "initiative"],
        "dislikes": ["tactical nitpicking", "short-sighted tone"]
    }
}


def classify_message_tone(message):
    tone_tags = []
    if any(word in message.lower() for word in ["blunt", "terse", "strong", "without giving context"]):
        tone_tags.append("direct/blunt")
    if any(word in message.lower() for word in ["emoji", "support"]):
        tone_tags.append("positive/reinforcing")
    if "asked" in message.lower():
        tone_tags.append("directive")
    return tone_tags or ["neutral"]


def simulate_persona_reactions(message):
    reactions = {}
    for persona, traits in PERSONAS.items():
        rapport_score = 7  # baseline score
        tone_tags = classify_message_tone(message)

        # Penalize or reward based on tone and persona preferences
        if "direct/blunt" in tone_tags:
            if "bluntness" in traits["dislikes"]:
                rapport_score -= 2
        if "positive/reinforcing" in tone_tags:
            if "positive tone" in traits["values"]:
                rapport_score += 2

        reactions[persona] = {
            "rapport_score": max(0, min(10, rapport_score)),
            "tags": tone_tags,
            "summary": f"As a {persona}, this message feels {'well-aligned' if rapport_score >= 7 else 'slightly off'}."
        }
    return reactions


def run_tone_analysis(messages):
    report = {}
    for msg in messages:
        report[msg] = simulate_persona_reactions(msg)
    return report


if __name__ == "__main__":
    results = run_tone_analysis(SAMPLE_MESSAGES)
    output_file = Path("tone_analysis_report.json")
    output_file.write_text(json.dumps(results, indent=2))
    print(f"Analysis complete. Results saved to {output_file.resolve()}")