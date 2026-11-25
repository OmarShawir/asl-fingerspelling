"""Offline CTC greedy collapse and edit metrics; not a streaming word builder."""

from fingerspelling.sequence_data import VOCABULARY


def greedy_decode(tokens):
    result, previous = [], None
    for token in tokens:
        token = int(token)
        if not 0 <= token <= len(VOCABULARY):
            raise ValueError("Unknown CTC token")
        if token and token != previous:
            result.append(VOCABULARY[token - 1])
        previous = token
    return "".join(result)


def edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def transcript_metrics(rows):
    total = len(rows)
    characters = sum(len(r["target"]) for r in rows)
    edits = sum(edit_distance(r["target"], r["prediction"]) for r in rows)
    exact = sum(r["target"] == r["prediction"] for r in rows)
    empty = sum(not r["prediction"] for r in rows)
    return dict(
        sequences=total,
        reference_characters=characters,
        character_edits=edits,
        cer=edits / characters if characters else None,
        exact=exact,
        exact_accuracy=exact / total if total else None,
        empty=empty,
        empty_rate=empty / total if total else None,
    )
