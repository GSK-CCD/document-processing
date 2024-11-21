from pathlib import Path
from typing import Type

from dora_ki.doc_processing.base import BaseFileProcessor
from dora_ki.doc_processing.pdfs import PdfProcessor
from dora_ki.doc_processing.word_docs import WordDocProcessor

file_type_processors = {"pdf": PdfProcessor, "docx": WordDocProcessor}


def file_processor_factory(file_path: str) -> Type[BaseFileProcessor]:
    file_type = Path(file_path).suffix[1:]
    return file_type_processors[file_type]

