"""File readers, loaded independently when their format is requested."""

from importlib import import_module

_MODULES = {
    "read_csv": "csv_reader",
    "read_directory": "directory_reader",
    "read_docx": "docx_reader",
    "read_json": "json_reader",
    "read_lines": "line_reader",
    "read_pdf": "pdf_reader",
    "search_file": "search_reader",
}

__all__ = [
    "read_directory",
    "read_lines",
    "search_file",
    "read_csv",
    "read_json",
    "read_pdf",
    "read_docx",
]


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
    globals()[name] = value
    return value
