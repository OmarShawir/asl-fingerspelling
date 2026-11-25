"""Pure checks shared by the dataset audit and its small regression tests."""
ALPHABET_WITH_SPACE = frozenset("abcdefghijklmnopqrstuvwxyz ")


def ctc_minimum(text: str) -> int:
    """CTC needs a blank between equal adjacent target symbols."""
    return len(text) + sum(a == b for a, b in zip(text, text[1:]))


def alphabet_scope(text: str) -> bool:
    """Keep a whole sequence or reject it; never delete target characters."""
    return bool(text) and set(text) <= ALPHABET_WITH_SPACE


def split_name(original: str) -> str:
    return "validation" if original == "dev" else original


def hand_blocks(frame):
    """Structural blocks only. Their anatomical identity/order is UNVERIFIED.

    Do not label block 0 'left' or 'right'. A future experimental set encoder
    may pool these blocks symmetrically, avoiding a fixed-order assumption.
    """
    if frame is None or len(frame) not in (0, 21, 42):
        raise ValueError("Expected zero, 21, or 42 points")
    if any(len(point) != 3 for point in frame):
        raise ValueError("Each point must have three coordinates")
    return [frame[i:i + 21] for i in range(0, len(frame), 21)]
