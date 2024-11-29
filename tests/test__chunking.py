import re
from typing import List

import pytest
from llama_index.core.schema import TextNode

from document_processing.chunking import ChunkMeta, Chunks, FunctionChunker


def splitter(text: str):
    return [TextNode(text=t) for t in re.split(r"\.|\?|!", text) if t]


@pytest.mark.parametrize(
    "chunks,expected",
    [
        (
            [TextNode(text="This is "), TextNode(text="a test"), TextNode(text=".")],
            [TextNode(text="This is a test.")],
        ),
        (
            [TextNode(text="This is a test")],
            [TextNode(text="This is a test")],
        ),
        (
            [TextNode(text="This is "), TextNode(text="a test."), TextNode(text="How many apples in a bunch?")],
            [TextNode(text="This is a test."), TextNode(text="How many apples in a bunch?")],
        ),
        (
            [TextNode(text="How many apples in a bunch? "), TextNode(text="This is "), TextNode(text="a test."), TextNode(text=" Goodbye.")],
            [TextNode(text="How many apples in a bunch? This is a test. Goodbye.")],
        ),
        (
            [
                TextNode(text="This is"),
                TextNode(text=" a much longer text coming after so add it to the previous."),
                TextNode(text="This is long enough to be alone"),
            ],
            [TextNode(text="This is a much longer text coming after so add it to the previous."), TextNode(text="This is long enough to be alone")],
        ),
    ],
)
def test__function_chunker_short_chunks(chunks: List[TextNode], expected: List[TextNode]):
    func_chunk = FunctionChunker(5, 80, splitter, chat_model="gpt-4o")
    new_chunks = func_chunk.combine_short_chunks(chunks)
    for chunk, expected_chunk in zip(new_chunks, expected):
        assert chunk.text == expected_chunk.text


@pytest.mark.parametrize(
    "chunks,expected",
    [
        (
            [TextNode(text="How many apples in a bunch? This is a test. Goodbye.")],
            [TextNode(text="How many apples in a bunch"), TextNode(text=" This is a test"), TextNode(text=" Goodbye")],
        ),
        (
            [TextNode(text="This is a test")],
            [TextNode(text="This is a test")],
        ),
        (
            [TextNode(text="This is "), TextNode(text="How many apples in a bunch? This is a test. Goodbye."), TextNode(text="a test.")],
            [
                TextNode(text="This is "),
                TextNode(text="How many apples in a bunch"),
                TextNode(text=" This is a test"),
                TextNode(text=" Goodbye"),
                TextNode(text="a test."),
            ],
        ),
    ],
)
def test__function_chunker_long_chunks(chunks: List[TextNode], expected: List[TextNode]):
    func_chunk = FunctionChunker(1, 10, splitter, chat_model="gpt-4o")
    new_chunks = func_chunk.split_large_chunks_down(chunks)
    for chunk, expected_chunk in zip(new_chunks, expected):
        assert chunk.text == expected_chunk.text


@pytest.mark.parametrize(
    "chunks,metas,expected",
    [
        (
            ["ghi", "abc", "def"],
            [
                ChunkMeta(chunk_number=2, length=3, lang="en"),
                ChunkMeta(chunk_number=0, length=3, lang="en"),
                ChunkMeta(chunk_number=1, length=3, lang="en"),
            ],
            ["abc", "def", "ghi"],
        ),
        (
            ["ghi", "abc", "def"],
            [
                ChunkMeta(chunk_number=3, length=3, lang="en"),
                ChunkMeta(chunk_number=2, length=3, lang="en"),
                ChunkMeta(chunk_number=2, length=3, lang="en"),
            ],
            ["abc", "def", "ghi"],
        ),
    ],
)
def test__order_chunks(chunks: List[str], metas: List[ChunkMeta], expected: List[str]):
    chunks_ = Chunks(chunks=chunks, metas=metas)
    new_chunks = chunks_.order_chunks()
    for chunk, expected_chunk in zip(new_chunks, expected):
        assert chunk == expected_chunk


@pytest.mark.parametrize(
    "chunks,metas,expected",
    [
        (
            ["ghi", "abc", "def"],
            [
                ChunkMeta(chunk_number=2, length=3, lang="en"),
                ChunkMeta(chunk_number=0, length=3, lang="en"),
                ChunkMeta(chunk_number=1, length=3, lang="en"),
            ],
            "[...] ghi [...]\n\n\n\n[...] abc [...]\n\n\n\n[...] def [...]",
        )
    ],
)
def test__join_chunks(chunks: List[str], metas: List[ChunkMeta], expected: List[str]):
    chunks_ = Chunks(chunks=chunks, metas=metas)
    joined_chunks = chunks_.join_self_chunks()
    assert joined_chunks == expected


@pytest.mark.parametrize(
    "text,n_spaces,from_start,expected",
    [
        ("This is a test", 1, True, "This is"),
        ("This is a test", 2, True, "This is a"),
        ("This is a test", 2, False, "is a test"),
        ("How many apples in a bunch", 3, True, "How many apples in"),
        ("How many apples in a bunch", 4, True, "How many apples in a"),
        ("How many apples in a bunch", 3, False, "apples in a bunch"),
    ],
)
def test__extract_context_from_chunk(text, n_spaces, from_start, expected):
    func_chunk = FunctionChunker(1, 10, splitter, chat_model="gpt-4o")
    context = func_chunk.extract_context_from_chunk(text, n_spaces, from_start=from_start)
    assert context == expected
