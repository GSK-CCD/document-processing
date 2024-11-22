from typing import Callable, List, cast

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

        nodes = cast(List[TextNode], self.sentence_splitter.get_nodes_from_documents([text_document]))

        return self.add_context(nodes, num_words_overlap)


class FunctionChunker(BaseChunker):
    """
    A chunker that uses any user defined function to split the text into chunks.
    """

    def __init__(self, min_length: int, max_length: int, splitting_function: Callable[[str], List[TextNode]]):
        self.min_length = min_length
        self.max_length = max_length
        self.splitting_function = splitting_function

    def chunk(self, text: str, num_words_overlap: int) -> List[TextNode]:
        chunks = self.splitting_function(text)
        chunks = self.ensure_chunks_large_enough(chunks)
        chunks = self.ensure_chunks_small_enough(chunks)
        return self.add_context(chunks, num_words_overlap)

    def ensure_chunks_large_enough(self, chunks):
        all_chunks_above_min = False
        while not all_chunks_above_min:
            all_chunks_above_min = all(len(chunk) >= self.min_length for chunk in chunks)
            chunks = self.combine_short_strings(chunks, self.min_length)
        return chunks

    def ensure_chunks_small_enough(self, chunks):
        all_chunks_below_max = False
        while not all_chunks_below_max:
            all_chunks_below_max = all(len(chunk) <= self.max_length for chunk in chunks)
            chunks = self.split_and_insert(chunks, self.max_length, self.splitting_function)
        return chunks

    def combine_short_strings(strings, min_length):
        """Combines chunks that are too short into the previous chunk."""
        result = []
        buffer = ""
        for string in strings:
            if len(string) < min_length:
                # Combine short strings with the buffer
                buffer += string
            else:
                # Append buffer to the result if it exists
                if buffer:
                    if result:
                        result[-1] += buffer
                    else:
                        result.append(buffer)
                    buffer = ""
                # Append the current string to the result
                result.append(string)
        # Append any remaining buffer to the last string in the result
        if buffer:
            if result:
                result[-1] += buffer
            else:
                result.append(buffer)
        return result

    def split_and_insert(li, max_size, splitting_function):
        """
        Splits chunks that are too long into smaller chunks using the provided splitting function.
        """
        new_li = []
        for item in li:
            if len(item) > max_size:
                # Split the long chunk into smaller parts
                split_parts = splitting_function(item)
                new_li.extend(split_parts)  # Add the split parts to the new list
            else:
                new_li.append(item)  # Add the chunk as is if it's within the size limit
        return new_li


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
