import logging
from pathlib import Path

import pymupdf  # type: ignore
import pymupdf4llm
from starlette import datastructures

from document_processing.base import BaseFileProcessor

logger = logging.getLogger(__name__)


class PdfProcessor(BaseFileProcessor):

    async def _get_doc(self):
        if isinstance(self.file_name, str):
            return pymupdf.open(self.file_name)
        elif isinstance(self.file_name, datastructures.UploadFile):
            file_type = Path(self.file_name.filename).suffix[1:]
            page = await self.file_name.read()
            pdf = pymupdf.open(stream=page, filetype=file_type)
            return pdf
        else:
            raise ValueError(f"file_name must be either a string or a datastructures.UploadFile, got '{type(self.file_name)}'")

    async def extract_text(self) -> str:
        """
        Takes a PDF file and extracts the text from it to a string.
        """
        doc = await self._get_doc()
        text_contents = ""
        for page in doc:
            text = page.get_text()
            text_contents += text
        return text_contents

    def extract_text_llm(self) -> str:
        md_text = pymupdf4llm.to_markdown(self.file_name)
        return md_text
