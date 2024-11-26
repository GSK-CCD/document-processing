from typing import List

from llama_index.core.evaluation import ContextRelevancyEvaluator


def llm_based_context_eval(model, query, contexts):
    evaluator = ContextRelevancyEvaluator(llm=model)
    evaluator.aevaluate(query=query, contexts=contexts, sleep_time_in_seconds=1.0)


def eval_rule_based(retrieved_context: List[str], expected_context: List[str]):

    pass


def check_overlaps(retrieved_context: List[str], expected_context: List[str]):
    overlap_results = []
    for expected in expected_context:
        overlap = get_overlap_single_context(retrieved_context, expected)
        overlap_results.append(overlap)


def get_overlap_single_context(retrieved_context: List[str], expected: str) -> float:
    overlap = 0.0
    joined_retrieved = "\n".join(retrieved_context)
    if expected not in joined_retrieved:
        return overlap
    replaced = joined_retrieved.replace(expected, "")
    n_overlap = len(joined_retrieved) - len(replaced)
    overlap = n_overlap / len(joined_retrieved)
    return overlap


def total_overlap(retrieved_context: List[str], expected_context: List[str], individual_overlaps: List[float]) -> float:
    total_length_retrieved = sum(len(context) for context in retrieved_context)
    detected_contexts = [context for context, overlap in zip(expected_context, individual_overlaps) if overlap > 0]
    total_length_detected = sum(len(context) for context in detected_contexts)
    return total_length_detected / total_length_retrieved
