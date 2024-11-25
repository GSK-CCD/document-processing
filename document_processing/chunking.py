from typing import Callable, List, cast

import langdetect
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document, TextNode
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding  # type: ignore
from pydantic import BaseModel  # type: ignore

from document_processing.base import BaseChunker
from document_processing.embeddings import check_n_embeddings
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

        nodes = cast(List[TextNode], self.sentence_splitter.get_nodes_from_documents([text_document]))

        return self.add_context(nodes, num_words_overlap)


class FunctionChunker(BaseChunker):
    """
    A chunker that uses any user defined function to split the text into chunks.
    """

    def __init__(self, min_length_tokens: int, max_length_tokens: int, splitting_function: Callable[[str], List[TextNode]], chat_model="gpt-4o"):
        self.min_length = min_length_tokens
        self.max_length = max_length_tokens
        self.splitting_function = splitting_function
        self.chat_model = chat_model

    def chunk(self, text: str, num_words_overlap: int) -> List[TextNode]:
        chunks = self.splitting_function(text)
        chunks = self.ensure_chunks_small_enough(chunks)
        chunks = self.ensure_chunks_large_enough(chunks)
        return self.add_context(chunks, num_words_overlap)

    def ensure_chunks_large_enough(self, chunks: List[TextNode]) -> List[TextNode]:
        all_chunks_above_min = False
        while True:
            all_chunks_above_min = all(check_n_embeddings(chunk.text, self.chat_model) >= self.min_length for chunk in chunks)
            if all_chunks_above_min:
                break
            chunks = self.combine_short_chunks(chunks)
        return chunks

    def ensure_chunks_small_enough(self, chunks):
        all_chunks_below_max = False
        while True:
            all_chunks_below_max = all(check_n_embeddings(chunk.text, self.chat_model) <= self.max_length for chunk in chunks)
            if all_chunks_below_max:
                break
            chunks = self.split_large_chunks_down(chunks)
        return chunks

    def combine_short_chunks(self, nodes: List[TextNode]) -> List[TextNode]:
        """Combines chunks that are too short into the previous chunk."""
        result: List[TextNode] = []
        buffer = ""
        for node in nodes:
            string = node.text
            if check_n_embeddings(string, self.chat_model) < self.min_length:
                # Combine short strings with the buffer
                buffer += string
            else:
                # Append buffer to the result if it exists
                if buffer:
                    if result:
                        last_node = result[-1]
                        last_node.text += buffer
                        result[-1] = last_node
                    else:
                        result.append(TextNode(text=buffer))
                    buffer = ""
                # Append the current string to the result
                result.append(TextNode(text=string))
        # Append any remaining buffer to the last string in the result
        if buffer:
            if result:
                last_node = result[-1]
                last_node.text += buffer
                result[-1] = last_node
            else:
                result.append(TextNode(text=buffer))
        result = self._add_metadata(result)
        return result

    def split_large_chunks_down(self, nodes: List[TextNode]):
        """
        Splits chunks that are too long into smaller chunks using the provided splitting function.
        """
        new_nodes = []
        for node in nodes:
            text = node.text
            if check_n_embeddings(text, self.chat_model) > self.max_length:
                split_parts = self.splitting_function(text)
                new_nodes.extend(split_parts)
            else:
                new_nodes.append(TextNode(text=text))
        new_nodes = self._add_metadata(new_nodes)
        return new_nodes

    def _add_metadata(self, chunks: List[TextNode]) -> List[TextNode]:
        return [
            TextNode(text=chunk.text, metadata=ChunkMeta(chunk_number=i, length=len(chunk.text), lang=langdetect.detect(chunk.text)).model_dump())
            for i, chunk in enumerate(chunks)
        ]


class ChunkMeta(BaseModel):
    chunk_number: int
    length: int
    lang: str


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
