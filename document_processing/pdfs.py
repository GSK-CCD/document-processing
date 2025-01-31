import logging

import pymupdf  # type: ignore
import pymupdf4llm

from document_processing.base import BaseFileProcessor

logger = logging.getLogger(__name__)


class PdfProcessor(BaseFileProcessor):

    def _get_doc(self):
        if isinstance(self.file_name, str):
            return pymupdf.open(self.file_name)
        else:
            raise ValueError(f"file_name must be a string, got '{type(self.file_name)}'")

    def extract_text(self) -> str:
        """
        Takes a PDF file and extracts the text from it to a string.
        """
        doc = self._get_doc()
        text_contents = ""
        for page in doc:
            text = page.get_text()
            text_contents += text
        return text_contents

    def extract_text_llm(self) -> str:
        md_text = pymupdf4llm.to_markdown(self.file_name)
        return md_text
