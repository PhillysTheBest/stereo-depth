from stereo_depth.api.openapi import DEFAULT_PATH, render


def test_committed_schema_is_current() -> None:
    # Regenerate with: uv run python -m stereo_depth.api.openapi
    assert DEFAULT_PATH.read_text() == render()
