"""Dataset loading and Wikipedia passage corpus construction for UA-ACD."""
import re
import random
from typing import List, Tuple, Optional

import numpy as np
from datasets import load_dataset


def load_factscore_topics(n_samples: int = 50, seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Load biography topics from the FactScore benchmark dataset.

    Returns:
        topics:       list of query strings ("Tell me a bio of X.")
        entity_names: list of entity name strings
    """
    ds = load_dataset("shmsw25/FActScoring", split="test")
    random.seed(seed)
    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))

    topics, entity_names = [], []
    for i in indices:
        row = ds[i]
        entity = row.get("title") or row.get("entity") or row.get("input", "")
        entity = entity.strip()
        topic = f"Tell me a bio of {entity}."
        topics.append(topic)
        entity_names.append(entity)

    return topics, entity_names


def load_wikipedia_passages(
    entity_names: List[str],
    max_passages_per_entity: int = 20,
) -> Tuple[List[str], List[str]]:
    """
    Build a Wikipedia passage corpus for the given entities.

    Returns:
        all_passages:      flat list of passage strings
        passage_to_entity: parallel list mapping each passage to its entity (lowercased)
    """
    wiki_ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

    entity_set = {e.lower() for e in entity_names}
    collected: dict = {e: [] for e in entity_set}
    needed = len(entity_set) * max_passages_per_entity

    for article in wiki_ds:
        title = article.get("title", "").lower()
        if title not in entity_set:
            continue
        text = article.get("text", "")
        # Split into ~100-word passages
        words = text.split()
        for start in range(0, min(len(words), max_passages_per_entity * 100), 100):
            chunk = " ".join(words[start:start + 100])
            if len(chunk.strip()) > 20:
                collected[title].append(chunk)
            if len(collected[title]) >= max_passages_per_entity:
                break
        if all(len(v) >= max_passages_per_entity for v in collected.values()):
            break

    all_passages, passage_to_entity = [], []
    for entity, passages in collected.items():
        for p in passages:
            all_passages.append(p)
            passage_to_entity.append(entity)

    return all_passages, passage_to_entity
