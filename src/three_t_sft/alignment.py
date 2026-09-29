"""Sparse overlap alignment in UTF-8 bytes of the complete receiver prompt."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def __post_init__(self):
        if self.start < 0 or self.end < self.start:
            raise ValueError('Expected nonnegative ordered byte offsets')


@dataclass(frozen=True)
class Alignment:
    receiver_indices: tuple[int, ...]
    sender_count: int
    # (local receiver row, sender column, normalized overlap)
    entries: tuple[tuple[int, int, float], ...]

    @property
    def rows(self):
        return len(self.receiver_indices)

    @property
    def valid_rows(self):
        return tuple(sorted({row for row, _, _ in self.entries}))

    def tensor(self, *, device=None, dtype=None):
        import torch
        indices = torch.tensor([(r, c) for r, c, _ in self.entries],
                               dtype=torch.long, device=device).reshape(-1, 2).T
        weights = torch.tensor([v for _, _, v in self.entries], dtype=dtype, device=device)
        return torch.sparse_coo_tensor(indices, weights, (self.rows, self.sender_count),
                                       check_invariants=True).coalesce()


def character_offsets_to_bytes(text, offsets):
    """Convert tokenizer character offsets against the EXACT input string."""
    prefix = [0]
    for char in text:
        prefix.append(prefix[-1] + len(char.encode('utf-8')))
    result = []
    for a, b in offsets:
        if not 0 <= a <= b <= len(text):
            raise ValueError('Tokenizer offset outside input text')
        result.append(Span(prefix[a], prefix[b]))
    return tuple(result)


def align(sender_spans, receiver_spans, message_span, *, special_indices=()):
    """Sender spans are message-local; receiver spans are prompt-global.

    Spans must form ordered, nonoverlapping byte intervals. Tokenizers with
    overlapping/ambiguous offsets need an adapter; we do not silently guess.
    Prompt boundary tokens are clipped to the message region before overlap.
    """
    def validate(spans):
        previous = 0
        for span in spans:
            if span.start < previous:
                raise ValueError('Overlapping or unsorted token spans')
            previous = span.end
    validate(sender_spans)
    # Special tokens may have (0,0) offsets interspersed with actual tokens.
    specials = set(special_indices)
    validate([s for i, s in enumerate(receiver_spans) if i not in specials and s.end > s.start])
    if any(s.end > message_span.end-message_span.start for s in sender_spans):
        raise ValueError('Sender span exceeds transmitted message')
    selected, entries, cursor = [], [], 0
    for index, span in enumerate(receiver_spans):
        a, b = max(span.start, message_span.start), min(span.end, message_span.end)
        if index in specials or b <= a:
            continue
        row = len(selected)
        selected.append(index)
        a -= message_span.start; b -= message_span.start
        while cursor < len(sender_spans) and sender_spans[cursor].end <= a:
            cursor += 1
        col, overlaps = cursor, []
        while col < len(sender_spans) and sender_spans[col].start < b:
            size = max(0, min(b, sender_spans[col].end)-max(a, sender_spans[col].start))
            if size: overlaps.append((col, size))
            col += 1
        total = sum(size for _, size in overlaps)
        entries.extend((row, col, size/total) for col, size in overlaps)
    return Alignment(tuple(selected), len(sender_spans), tuple(entries))
