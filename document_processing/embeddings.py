import tiktoken


def check_n_embeddings(text: str, embedding_model_name: str) -> int:
    encoding = tiktoken.encoding_for_model(embedding_model_name)
    return len(encoding.encode(text))

