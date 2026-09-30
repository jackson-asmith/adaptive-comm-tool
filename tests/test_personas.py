import pytest

from adaptive_comm.personas import PersonaFileError, load_personas


def test_default_personas_load(personas):
    assert [p.name for p in personas] == ["pragmatic_engineer", "empathic_pm", "vision_director"]
    assert all(p.values and p.dislikes for p in personas)


def test_custom_personas_file(tmp_path):
    f = tmp_path / "team.yaml"
    f.write_text("alex:\n  role: Designer\n  values: [craft]\n  dislikes: [rushing]\n")
    [alex] = load_personas(f)
    assert alex.name == "alex" and alex.role == "Designer" and alex.description == ""


@pytest.mark.parametrize(
    "content",
    [
        "",  # empty
        "- just a list",  # not a mapping
        "bob: not-a-mapping",  # persona isn't a mapping
        "bob:\n  role: X\n  values: [a]\n",  # missing dislikes
        "bob:\n  role: X\n  values: [a]\n  dislikes: [b]\n  typo: 1\n",  # unknown field
        "bob: [unclosed",  # invalid YAML
    ],
)
def test_bad_personas_files_rejected(tmp_path, content):
    f = tmp_path / "bad.yaml"
    f.write_text(content)
    with pytest.raises(PersonaFileError):
        load_personas(f)


def test_missing_personas_file(tmp_path):
    with pytest.raises(PersonaFileError, match="not found"):
        load_personas(tmp_path / "nope.yaml")
