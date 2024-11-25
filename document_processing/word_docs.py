from docx import Document
from docx.oxml.table import CT_Tbl as table_type
from docx.oxml.text.paragraph import CT_P as paragraph_type
from docx.table import Table

from document_processing.base import BaseFileProcessor


class WordDocProcessor(BaseFileProcessor):

    def extract_text(self, target_column: int = 0) -> str:
        """
        Takes a PDF file and extracts the text from it to a string.
        """
        doc = Document(self.file_name)
        full_text = ""
        tables = doc.tables
        for element in doc.element.body:
            if isinstance(element, paragraph_type):
                full_text += "\n" + element.text
            elif isinstance(element, table_type):
                full_text += "\n" + self._extract_from_table(tables.pop(0), target_column)
        return full_text

    def _extract_from_table(self, table: Table, target_column: int) -> str:
        table_text = ""
        for row in table.rows:
            if len(row.cells) == 1:
                table_cell = row.cells[0]
            else:
                table_cell = row.cells[target_column]
            table_text += table_cell.text
        return table_text
