"""One argument contract for file writes and their UI previews."""


def write_text(command: str, file_text=None, old_str="", new_str=None, insert_line=0) -> str:
    """Resolve the intended text without discarding a supplied payload."""
    if command not in {"create", "str_replace", "insert", "append"}:
        raise ValueError("Invalid command. Use create, str_replace, insert or append.")
    for name, value in (("file_text", file_text), ("old_str", old_str), ("new_str", new_str)):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{name} must be text.")
    if command == "create":
        if old_str or new_str not in (None, "") or insert_line:
            raise ValueError("create uses file_text; replacement and insertion arguments are not accepted.")
        return file_text or ""  # Creating an explicitly empty file remains supported.
    if command == "str_replace":
        if file_text not in (None, "") or insert_line:
            raise ValueError("str_replace uses old_str and new_str, not file_text or insert_line.")
        if not old_str or new_str is None:
            raise ValueError("str_replace requires nonempty old_str and explicit new_str (use \"\" to delete).")
        if old_str == new_str:
            raise ValueError("No changes to apply: old_str and new_str are identical.")
        return new_str
    if old_str or (command == "append" and insert_line):
        raise ValueError(f"{command} does not use old_str" + (" or insert_line." if command == "append" else "."))
    if command == "append" and new_str == "" and file_text:
        new_str = None  # an empty append is not a competing valid payload
    if new_str is not None and file_text is not None and new_str != file_text:
        raise ValueError(f"{command} received conflicting new_str and file_text. Supply one text payload.")
    text = new_str if new_str is not None else file_text
    if text is None or (command == "append" and text == ""):
        raise ValueError(f"{command} requires " + ("nonempty " if command == "append" else "") +
                         "new_str (file_text is also accepted). No file was changed.")
    return text
