from abc import ABC, abstractmethod
from typing import List

from llama_index.core.schema import TextNode


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text, num_words_overlap: int) -> List[TextNode]:
        pass


class BaseFileProcessor:
    def __init__(self, file_name: str, chunker: BaseChunker):
        self.file_name = file_name
        self.chunker = chunker

    @abstractmethod
    def extract_text(self) -> str:
        raise NotImplementedError

    def chunk(self, text: str, num_words_overlap: int) -> List[TextNode]:
        return self.chunker.chunk(text, num_words_overlap)

