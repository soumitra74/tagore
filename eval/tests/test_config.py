from config import Settings


def test_settings_defaults_when_no_env_file_present():
    # _env_file=None skips reading the repo-root .env so this test's
    # result doesn't depend on what's actually in that (gitignored) file.
    settings = Settings(_env_file=None)
    assert settings.eval_judge_provider == "openai"
    assert settings.eval_judge_model == "gpt-4o-mini"
