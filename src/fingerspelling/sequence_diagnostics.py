"""Deterministic edit attribution; never modifies CTC targets or decoding."""

from collections import Counter

from fingerspelling.ctc_decode import transcript_metrics

TIE_RULE = (
    "Unit-cost Levenshtein DP; traceback from the end prefers equal diagonal match, "
    "then diagonal substitution, then reference deletion, then hypothesis insertion. "
    "Counts on ambiguous alignments depend on this deterministic rule."
)


def align(reference, hypothesis):
    dp = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    for i in range(len(reference) + 1):
        dp[i][0] = i
    for j in range(len(hypothesis) + 1):
        dp[0][j] = j
    for i, a in enumerate(reference, 1):
        for j, b in enumerate(hypothesis, 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (a != b))
    i, j = len(reference), len(hypothesis)
    result = []
    while i or j:
        if i and j and reference[i - 1] == hypothesis[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            result.append(("match", reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif i and j and dp[i][j] == dp[i - 1][j - 1] + 1:
            result.append(("substitution", reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            result.append(("deletion", reference[i - 1], ""))
            i -= 1
        else:
            result.append(("insertion", "", hypothesis[j - 1]))
            j -= 1
    return result[::-1]


def detailed_metrics(rows):
    metrics = transcript_metrics(rows)
    operations = Counter()
    matched = Counter()
    targets = Counter()
    emitted = Counter()
    per_clip = []
    for row in rows:
        alignment = align(row["target"], row["prediction"])
        counts = Counter(x[0] for x in alignment)
        operations.update(counts)
        matched.update(a for op, a, b in alignment if op == "match")
        targets.update(row["target"])
        emitted.update(row["prediction"])
        per_clip.append(
            dict(
                id=row["id"],
                target=row["target"],
                prediction=row["prediction"],
                deletions=counts["deletion"],
                substitutions=counts["substitution"],
                insertions=counts["insertion"],
            )
        )
    errors = sum(operations[k] for k in ("deletion", "substitution", "insertion"))
    assert errors == metrics["character_edits"]
    assert (
        sum(emitted.values()) - sum(targets.values())
        == operations["insertion"] - operations["deletion"]
    )
    metrics.update(
        deletions=operations["deletion"],
        substitutions=operations["substitution"],
        insertions=operations["insertion"],
        predicted_characters=sum(emitted.values()),
        target_counts=dict(sorted(targets.items())),
        output_counts=dict(sorted(emitted.items())),
        aligned_correct_counts=dict(sorted(matched.items())),
        alignment_tie_rule=TIE_RULE,
        focus={
            c: dict(target=targets[c], emitted=emitted[c], aligned_correct=matched[c])
            for c in "jzq "
        },
    )
    return metrics, per_clip
