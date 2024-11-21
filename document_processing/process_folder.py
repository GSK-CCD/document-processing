import pathlib
from typing import List


class FolderProcessing:
    def __init__(self, folder_name: str, file_pattern: str):
        self.folder_name = folder_name
        # get all pdf files in folder
        self.files = self.find_files(file_pattern)

    def find_files(self, file_pattern: str) -> List[pathlib.Path]:
        folder = pathlib.Path(self.folder_name)
        files = list(folder.glob(file_pattern))
        return files

