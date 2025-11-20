"""File readers package."""

from .directory_reader import read_directory
from .line_reader import read_lines
from .search_reader import search_file
from .csv_reader import read_csv
from .json_reader import read_json
from .pdf_reader import read_pdf
from .docx_reader import read_docx

__all__ = [
    "read_directory",
    "read_lines",
    "search_file",
    "read_csv",
    "read_json",
    "read_pdf",
    "read_docx",
]
