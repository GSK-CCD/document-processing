# Document Processing

This is a rpject which was priginally created with the intention of being re-usable code to standardize the way that we
process files like PDFs and word files. In the end it was mostly abandoned (although it is still used in a couple of
the earlier projects we worked on) and is thus not all that important.

The repo reads files and performs chunking. One of the chunking methods developed here in the repo uses an LLM to generate a
regex expression which can be used to split a document into it's main sections. For example, some files use chapter names or numbered titles 
for new sections, while others may use symbols in the section titles. This is why using an LLM seemed like a good fit. The LLM will read a couple pages of the document, and it should then figure out which convention a given file is using for section or chapter titles. The LLM
then tries to generate a valid regex expression which will split the text based on these sections. 

# Install

```bash
poetry install
```

# Usage

## Choose file processor

First step is to choose a file processor. We currently support two types: `PdfProcessor` and `WordDocXFileProcessor`.
If you are working with a mix of documents that might be PDF or Word files, you can use the `file_processor_factory` function for each file.
This will choose either the `PdfProcessor` or `WordDocXFileProcessor` based on the file extension.

```python
from document_processing.factory import file_processor_factory

file_processor = file_processor_factory(file_path)
```

With the `file_processor` you will be able to instantiate either the `PdfProcessor` or `WordDocXFileProcessor` class. Here you will need to
provide a file which can be either a string path to the file or a bytes object. If the file is a bytes object, it is assumed the bytes are a
stream and will thus be read from the stream. The second argument is the chunker. See the Chunking section below for information on how to 
choose a chunker.

```python
chunker = SemanticChunker(embedding_model, MAXIMUM_CHUNK_SIZE_IN_TOKENS, splitter_agent)
processor = file_processor(file, chunker=chunker)
```

After instantiating the processor, you can use it to extract text from the file and chunk the text into overlapping chunks. The `extract_text()`
will extract the entire text from the document into one string, and the chunker will then split the text into overlapping chunks.

```python
text = processor.extract_text()
chunks = processor.chunk(text, num_words_overlap=10)
```

## Chunking

There are two types of chunkers: `FunctionChunker` and `SemanticChunker`. You can also define your own by implementing the `BaseChunker` interface.

### FunctionChunker

The `FunctionChunker` is a chunker that splits the text into chunks based on a user defined function. The function can involve any sort of splitting logic, but it must return a list of `TextNode` objects.

```python
def splitter(text: str) -> List[TextNode]:
    return [TextNode(t) for t in text.split("\n\n")]

min_chunk_size = 50
max_chunk_size = 200
chunker = FunctionChunker(min_chunk_size, max_chunk_size, splitter)
```

### SemanticChunker

The `SemanticChunker` is a chunker that uses a semantic similarity score to determine where to split the text into chunks. The chunker uses an embedding
model to generate embeddings for sections of text, and then uses semantic similarity scores to determine where to split the text into chunks. The 
sections of text are split into sentences, and then a number of sentences are combined into a single chunk (the `buffer_size` parameter). Those 
sentences are then combined and compared to the next chunk of sentences. If the similarity score is below a certain threshold, the sentences are combined. The threshold is defined by the `breakpoint_percentile_threshold` parameter.

```python
embeddings = AzureOpenAIEmbedding()
n_sentences_to_combine = 2
threshold = 95
chunker = SemanticChunker(embeddings, n_sentences_to_combine, threshold)
```
