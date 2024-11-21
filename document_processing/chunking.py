from typing import List, cast

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document, TextNode
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding  # type: ignore
from pydantic import BaseModel  # type: ignore

from document_processing.base import BaseChunker
from document_processing.factory import file_processor_factory
from document_processing.word_docs import WordDocProcessor


class SemanticChunker(BaseChunker):

    def __init__(
        self,
        embed_model: AzureOpenAIEmbedding,
        buffer_size: int = 2,
        breakpoint_percentile_threshold: int = 95,
    ):
        self.embed_model = embed_model
        self.sentence_splitter = SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            embed_model=embed_model,
        )

    def chunk(self, text: str, num_words_overlap: int) -> List[TextNode]:
        """Chunks text into overlapping chunks."""

        text_document = Document(text=text)

        nodes = cast(
            List[TextNode],
            self.sentence_splitter.get_nodes_from_documents([text_document]),
        )

        new_nodes = []
        for i in range(len(nodes)):
            node = nodes[i]
            if num_words_overlap:
                previous_node = None
                next_node = None
                if i > 0:
                    previous_node = nodes[i - 1]
                if i < len(nodes) - 1:
                    next_node = nodes[i + 1]
                new_node = TextNode(text=self.add_context(node, previous_node, next_node, num_words_overlap))
            else:
                new_node = TextNode(text=node.text)
            new_nodes.append(new_node)
        return new_nodes

    def extract_text(self, text: str, num_spaces: int, from_start: bool = False):
        """Extracts text from a string, either from the start or the end. Used to get some overlapping
        text from the surrounding chunks. Breaks the original text up at each space in the sentence
        and then joins the words back together based on the number of spaces requested.
        """
        words = text.split()
        if from_start:
            return " ".join(words[: num_spaces + 1])
        else:
            return " ".join(words[-num_spaces:])

    def add_context(self, node, previous_node, next_node, n_words: int) -> str:
        """Adds some overlapping text from the previous and next chunks to the current chunk."""
        text = node.text
        n_spaces = n_words - 1
        if previous_node is not None:
            previous_text = self.extract_text(previous_node.text, n_spaces)
            text = f"{previous_text}\n\n{text}"
        if next_node is not None:
            next_text = self.extract_text(next_node.text, n_spaces, from_start=True)
            text = f"{text}\n\n{next_text}"

        return text


class ChunkMeta(BaseModel):
    chunk_number: int


class Chunks(BaseModel):
    chunks: List[str]
    metas: List[ChunkMeta]

    def order_chunks(self):
        """Orders chunks based on their chunk number metadata."""
        combined_meta_and_chunk = [{meta.chunk_number: chunk} for meta, chunk in zip(self.metas, self.chunks)]
        # sort the dictionaries based on the chunk number key
        sorted_meta_and_chunk = sorted(combined_meta_and_chunk, key=lambda x: next(iter(x)))
        # get the chuynk strings out of the sorted dictionaries
        ordered_chunks = [list(entry.values())[0] for entry in sorted_meta_and_chunk]
        return ordered_chunks

    def join_chunks(self, chunks: List[str]):
        return "\n *** END OF CHUNK *** \n".join([f"[...] {chunk} [...]" for chunk in chunks])

    def join_self_chunks(self):
        return self.join_chunks(self.chunks)

    def pre_process(self):
        """Processes the chunks by ordering them and joining them into a single string."""
        ordered_chunks = self.order_chunks()
        joined_chunks = self.join_chunks(ordered_chunks)
        return joined_chunks

    def pop_chunk(self):
        """Removes the last chunk and it's metadata from the list."""
        self.chunks.pop()
        self.metas.pop()


def chunk_input_file(file, embedding_model, n_overlap, text_extraction_kwargs):
    """Takes an input file and chunks it into overlapping chunks which can be added to a vector DB."""
    chunker = SemanticChunker(embed_model=embedding_model)
    file_processor = file_processor_factory(file)
    processor = file_processor(file, chunker=chunker)
    text = processor.extract_text(**text_extraction_kwargs if isinstance(processor, WordDocProcessor) else {})
    text_nodes = processor.chunk(text, num_words_overlap=n_overlap)
    return text_nodes
