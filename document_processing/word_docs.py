from typing import Optional

from docx import Document
from docx2python import docx2python
from docx.oxml.table import CT_Tbl as table_type
from docx.oxml.text.paragraph import CT_P as paragraph_type
from docx.table import Table

from document_processing.base import BaseFileProcessor


class WordDocProcessor(BaseFileProcessor):

    def extract_text(self, target_column: Optional[int] = None) -> str:
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
                full_text += "\n" + self.extract_from_table(tables.pop(0), target_column)
        return full_text

    def extract_from_table(self, table: Table, target_column: Optional[int]) -> str:
        if target_column is not None:
            return self._extract_specific_column(table, target_column)
        return self._table_to_markdown(table)

    def _extract_specific_column(self, table: Table, target_column: int) -> str:
        table_text = ""
        for row in table.rows:
            if len(row.cells) == 1:
                table_cell = row.cells[0]
            else:
                table_cell = row.cells[target_column]
            table_text += table_cell.text
        return table_text

    def _table_to_markdown(self, table: Table) -> str:
        table_text = ""
        for i, row in enumerate(table.rows):
            for j, cell in enumerate(row.cells):
                table_text += f"| {cell.text} "
                if j == len(row.cells) - 1:
                    table_text += "|\n"
            if i == 0:
                table_text += "| --- " * len(row.cells) + "|\n"
        return table_text


class WordDocXFileProcessor(BaseFileProcessor):

    def extract_text(self):
        document = docx2python(self.file_name)
        output_text = ""
        for string_section in self.get_strings_from_list(document.body):
            output_text += string_section
        return output_text

    def get_strings_from_list(self, _list: list):
        for item in _list:
            if isinstance(item, list):
                yield from self.get_strings_from_list(item)
            else:
                yield item
