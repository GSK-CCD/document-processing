from abc import ABC, abstractmethod
from typing import List, Optional

from llama_index.core.schema import TextNode


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text, num_words_overlap: int) -> List[TextNode]:
        pass

    @abstractmethod
    def achunk(self, text: str, num_words_overlap: int):
        pass

    def add_context(self, nodes: List[TextNode], num_words_overlap: int) -> List[TextNode]:
        """Takes some overlapping text from the previous and next chunks and adds it to the current chunk."""
        new_nodes = []
        for i, node in enumerate(nodes):
            if num_words_overlap:
                previous_node = None
                next_node = None
                if i > 0:
                    previous_node = nodes[i - 1]
                if i < len(nodes) - 1:
                    next_node = nodes[i + 1]
                new_node = TextNode(text=self._add_context(node, previous_node, next_node, num_words_overlap))
            else:
                new_node = TextNode(text=node.text)
            new_nodes.append(new_node)
        return new_nodes

    def extract_context_from_chunk(self, text: str, num_spaces: int, from_start: bool = False):
        """Extracts text from a chunk, either from the start or the end. Used to get some overlapping
        text from the surrounding chunks. Breaks the original text up at each space in the sentence
        and then joins the words back together based on the number of spaces requested.
        """
        words = text.split(" ")
        if from_start:
            return " ".join(words[: num_spaces + 1])
        else:
            return " ".join(words[-num_spaces - 1 :])

    def _add_context(self, node: TextNode, previous_node: Optional[TextNode], next_node: Optional[TextNode], n_words: int) -> str:
        """Adds some overlapping text from the previous and next chunks to the current chunk."""
        text = node.text
        n_spaces = n_words - 1
        if previous_node is not None:
            previous_text = self.extract_context_from_chunk(previous_node.text, n_spaces)
            text = f"{previous_text}\n\n{text}"
        if next_node is not None:
            next_text = self.extract_context_from_chunk(next_node.text, n_spaces, from_start=True)
            text = f"{text}\n\n{next_text}"

        return text


class BaseFileProcessor:
    def __init__(self, file_name: str, chunker: BaseChunker):
        self.file_name = file_name
        self.chunker = chunker

    @abstractmethod
    def extract_text(self) -> str:
        raise NotImplementedError

    def chunk(self, text: str, num_words_overlap: int) -> List[TextNode]:
        return self.chunker.chunk(text, num_words_overlap)

    def achunk(self, text: str, num_words_overlap: int):
        return self.chunker.achunk(text, num_words_overlap)
