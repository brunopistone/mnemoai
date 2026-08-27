"""Unit tests for the pinned footer line (client/ui/status_bar.py).

Pure formatting + layout, so everything here is asserted on text: the model name
shortener, the compact token count, the meter/threshold colors, and the layout's
one hard promise — never wider than the terminal, and the context readout flush
right so the number sits in the same column every repaint.
"""

import os

from mnemoai.client.ui import status_bar as sb


def _plain(segments) -> str:
    """The footer as it appears on screen (styles dropped)."""
    return "".join(text for _, text in segments)


class TestShortModelName:
    def test_drops_provider_prefix_and_version_tail(self):
        assert sb.short_model_name(
            "us.anthropic.claude-opus-5-20260514-v1:0"
        ) == "claude-opus-5"

    def test_keeps_a_version_like_dotted_tail(self):
        # `llama3.1:8b` is the model's own version, not a provider prefix.
        assert sb.short_model_name("llama3.1:8b") == "llama3.1:8b"

    def test_drops_a_path_prefix(self):
        assert sb.short_model_name("openai/gpt-5-mini") == "gpt-5-mini"

    def test_truncates_past_the_limit(self):
        out = sb.short_model_name("a" * 40, limit=10)
        assert len(out) == 10 and out.endswith("…")

    def test_empty_is_empty(self):
        assert sb.short_model_name("") == ""
        assert sb.short_model_name(None) == ""


class TestFormatTokens:
    def test_thresholds(self):
        assert sb.format_tokens(0) == "0"
        assert sb.format_tokens(512) == "512"
        assert sb.format_tokens(999) == "999"
        assert sb.format_tokens(90096) == "90.1k"
        assert sb.format_tokens(1166221) == "1.17M"

    def test_negative_and_none_floor_at_zero(self):
        assert sb.format_tokens(-5) == "0"
        assert sb.format_tokens(None) == "0"


class TestMeter:
    def test_length_is_fixed(self):
        for fraction in (0, 0.01, 0.5, 1, 5):
            assert len(sb.meter(fraction)) == 8

    def test_any_usage_shows_at_least_one_cell(self):
        assert sb.meter(0.001).startswith("▓")
        assert sb.meter(0) == "░" * 8

    def test_full_never_overflows(self):
        assert sb.meter(1.0) == "▓" * 8
        assert sb.meter(3.0) == "▓" * 8


class TestLevel:
    def test_bands(self):
        assert sb.level(0.0) == sb.OK
        assert sb.level(0.69) == sb.OK
        assert sb.level(0.70) == sb.WARN
        assert sb.level(0.89) == sb.WARN
        assert sb.level(0.90) == sb.CRIT
        assert sb.level(1.5) == sb.CRIT


class TestHomePath:
    def test_home_becomes_tilde(self):
        home = os.path.expanduser("~")
        assert sb.home_path(os.path.join(home, "dev", "x")) == os.path.join(
            "~", "dev", "x"
        )

    def test_outside_home_is_untouched(self):
        assert sb.home_path("/tmp/x") == "/tmp/x"


class TestContextGroup:
    def test_zero_is_a_dash(self):
        text, cls = sb.context_group(0, 1000)
        assert text == "—" and cls == sb.OK

    def test_meter_count_and_percent(self):
        text, cls = sb.context_group(90096, 1_000_000)
        assert "90.1k" in text and "9%" in text
        assert "▓" in text and "░" in text
        assert cls == sb.OK

    def test_estimate_is_marked(self):
        text, _ = sb.context_group(90096, 1_000_000, estimated=True)
        assert "~90.1k" in text

    def test_colors_follow_the_fill(self):
        assert sb.context_group(750_000, 1_000_000)[1] == sb.WARN
        assert sb.context_group(950_000, 1_000_000)[1] == sb.CRIT

    def test_tiny_share_floors_at_one_percent(self):
        text, _ = sb.context_group(500, 1_000_000)
        assert "<1%" in text

    def test_without_a_window_only_the_count(self):
        text, cls = sb.context_group(1234, 0)
        assert text == "1.2k" and cls == sb.OK


class TestSegments:
    def _line(self, **kw):
        kw.setdefault("model", "us.anthropic.claude-opus-5-20260514-v1:0")
        kw.setdefault("provider", "bedrock")
        kw.setdefault("cwd", "/Users/x/dev/mnemoai")
        kw.setdefault("tokens", 90096)
        kw.setdefault("window", 1_000_000)
        return _plain(sb.segments(**kw))

    def test_shows_model_provider_dir_and_readout(self):
        line = self._line(width=100)
        assert "claude-opus-5" in line
        assert "bedrock" in line
        assert "mnemoai" in line
        assert "90.1k" in line and "9%" in line

    def test_never_exceeds_the_width(self):
        for width in (10, 20, 30, 40, 60, 80, 120, 200):
            assert len(self._line(width=width)) <= width

    def test_readout_is_flush_right(self):
        for width in (60, 80, 120):
            assert len(self._line(width=width)) == width

    def test_model_carries_its_own_style(self):
        segments = sb.segments(
            model="llama3.1:8b", provider="ollama", cwd="/tmp/p",
            tokens=10, window=1000, width=80,
        )
        model_text = "".join(t for cls, t in segments if cls == sb.MODEL)
        assert model_text == "llama3.1:8b"

    def test_narrow_terminal_drops_groups_before_wrapping(self):
        narrow = self._line(width=28)
        assert "\n" not in narrow
        assert "90.1k" in narrow  # the readout is what survives

    def test_width_narrower_than_the_readout_still_fits(self):
        line = self._line(width=6)
        assert len(line) <= 6 and "\n" not in line

    def test_no_model_or_dir_still_renders(self):
        line = self._line(model="", provider="", cwd="", width=40)
        assert "90.1k" in line

    def test_critical_fill_styles_the_readout(self):
        segments = sb.segments(
            model="m", provider="p", cwd="/tmp", tokens=990, window=1000, width=80
        )
        assert any(cls == sb.CRIT for cls, _ in segments)
