import logging

import pymupdf  # type: ignore

from dora_ki.doc_processing.base import BaseFileProcessor

logger = logging.getLogger(__name__)


class PdfProcessor(BaseFileProcessor):

    def extract_text(self) -> str:
        """
        Takes a PDF file and extracts the text from it to a string.
        """
        doc = pymupdf.open(self.file_name)
        text_contents = ""
        for page in doc:
            text = page.get_text()
            text_contents += text
        # if len(text_contents) < 500:
        #     print("PDF text is below 500 characters, likely a scanned PDF, trying with OCR")
        #     text_contents = self.extract_from_image(doc)
        return text_contents

    # def extract_from_image(self, doc):
    #     text_contents = ""
    #     for page in doc:
    #         text_page = page.get_textpage_ocr(language="eng+deu")
    #         text = text_page.extractTEXT()
    #         text_contents += text
    #     return text_contents

