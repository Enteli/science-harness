import pytest


@pytest.fixture(autouse=True)
def isolate_experiment_journal(tmp_path, monkeypatch):
    """Test CLI invocations must never enter the operator's scientific history."""
    monkeypatch.setenv("SCIENCE_HARNESS_HISTORY_DIR", str(tmp_path / "test-history"))


@pytest.fixture(autouse=True)
def require_stubbed_provider_http(monkeypatch):
    from science_harness import llm

    def blocked(*args, **kwargs):
        raise AssertionError("Tests must stub provider HTTP before requesting inference")

    monkeypatch.setattr(llm.urllib.request, "urlopen", blocked)
    for name in ("SCIENCE_HARNESS_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
