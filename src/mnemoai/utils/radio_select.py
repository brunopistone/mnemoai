"""One fact about ``prompt_toolkit``'s ``RadioList``: which row it is HIGHLIGHTING.

A ``RadioList`` tracks the highlighted row (``_selected_index``) separately from
its committed ``current_value``, and ``select_on_focus=True`` reconciles the two
in only SOME of its own bindings: ``up``/``down`` and the number keys commit,
while ``pageup``/``pagedown`` and type-to-find move the index and leave
``current_value`` on the row it was on before. Every dialog here overrides Enter
to confirm the row directly (no Tab-to-OK step), so a caller reading
``current_value`` answers with the row the user was on BEFORE they paged — PgDn
then Enter opened the wrong conversation, the same defect ``select_on_focus``
was added to fix for the arrow keys.

:func:`highlighted_value` is therefore what a confirm reads: derived from the
highlight, so it is right for every key that moves it, including one upstream
adds later. :func:`commit_paging` closes the visible half — it keeps the ``(*)``
marker on the row being paged to, so what the dialog shows and what it returns
cannot disagree.
"""


def highlighted_value(radio):
    """The value of the row ``radio`` is highlighting, not its stale commit."""
    try:
        return radio.values[radio._selected_index][0]
    except Exception:
        # A widget shape we don't recognize: the committed value is the best
        # answer left, and a picker must always return one.
        return radio.current_value


def commit_paging(radio) -> None:
    """Make PgUp/PgDn commit the row they land on, as ↑/↓ already do.

    Re-states the upstream move because a key resolves to exactly ONE handler
    (``KeyProcessor`` calls ``matches[-1]``), so there is no way to run alongside
    the widget's own binding — only to replace it.
    """

    def _move(event, sign: int) -> None:
        try:
            info = event.app.layout.current_window.render_info
            step = len(info.displayed_lines) if info else 1
            last = len(radio.values) - 1
            radio._selected_index = max(
                0, min(last, radio._selected_index + sign * step)
            )
            radio.current_value = radio.values[radio._selected_index][0]
        except Exception:
            # Paging is a convenience; a dialog must not die because it failed.
            pass

    radio.control.key_bindings.add("pageup")(lambda event: _move(event, -1))
    radio.control.key_bindings.add("pagedown")(lambda event: _move(event, 1))
