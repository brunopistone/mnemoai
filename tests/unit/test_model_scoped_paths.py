"""Unit tests for the model-id path sanitizer used to scope memory by model.

Episodic memory and the ACE playbook live under
``~/agent-conversations/{profile}/models/{sanitized_model}/`` so switching the
chat model doesn't contaminate memory built with a different one.
"""

from client.client import LangGraphClient


class TestSanitizeForPath:
    def test_slashes_and_colons_become_underscore(self):
        assert (
            LangGraphClient._sanitize_for_path("brnpistone/Qwen3.5-4B:latest")
            == "brnpistone_Qwen3.5-4B_latest"
        )

    def test_dotted_bedrock_id_preserved(self):
        # Dots, hyphens and digits are filesystem-safe and kept as-is.
        assert (
            LangGraphClient._sanitize_for_path("global.anthropic.claude-fable-5")
            == "global.anthropic.claude-fable-5"
        )

    def test_mantle_bare_id_preserved(self):
        assert LangGraphClient._sanitize_for_path("qwen.qwen3-32b") == "qwen.qwen3-32b"

    def test_spaces_collapsed(self):
        assert LangGraphClient._sanitize_for_path("a b  c") == "a_b_c"

    def test_empty_falls_back_to_default(self):
        assert LangGraphClient._sanitize_for_path("") == "default"

    def test_none_falls_back_to_default(self):
        assert LangGraphClient._sanitize_for_path(None) == "default"

    def test_result_is_single_path_segment(self):
        # Must never contain a path separator (would create nested dirs).
        out = LangGraphClient._sanitize_for_path("a/b/c:d e")
        assert "/" not in out and ":" not in out and " " not in out
