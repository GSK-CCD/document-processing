from pathlib import Path
from typing import Type

from document_processing.base import BaseFileProcessor
from document_processing.pdfs import PdfProcessor
from document_processing.word_docs import WordDocXFileProcessor

file_type_processors = {"pdf": PdfProcessor, "docx": WordDocXFileProcessor}


def file_processor_factory(file_path: str) -> Type[BaseFileProcessor]:
    file_type = Path(file_path).suffix[1:]
    return file_type_processors[file_type]
