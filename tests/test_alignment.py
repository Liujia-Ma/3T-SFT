import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from three_t_sft.alignment import Span, align, character_offsets_to_bytes


class AlignmentTests(unittest.TestCase):
    def test_utf8_not_character_length(self):
        spans = character_offsets_to_bytes('a中🙂', [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(spans, (Span(0, 1), Span(1, 4), Span(4, 8)))

    def test_boundary_clip_and_special_mask(self):
        a = align((Span(0, 2), Span(2, 5)),
                  (Span(0, 0), Span(0, 4), Span(4, 9), Span(0, 0)),
                  Span(2, 7), special_indices=(0, 3))
        self.assertEqual(a.receiver_indices, (1, 2))
        self.assertEqual(a.entries, ((0, 0, 1.), (1, 1, 1.)))

    def test_weighted_overlap(self):
        a = align((Span(0, 1), Span(1, 4)), (Span(0, 4),), Span(0, 4))
        self.assertEqual(a.entries, ((0, 0, .25), (0, 1, .75)))

    def test_zero_overlap_and_empty_message(self):
        a = align((Span(2, 3),), (Span(0, 1),), Span(0, 3))
        self.assertEqual((a.rows, a.valid_rows), (1, ()))
        self.assertEqual(align((), (), Span(0, 0)).rows, 0)

    def test_ambiguous_offsets_fail(self):
        with self.assertRaises(ValueError):
            align((Span(0, 3), Span(2, 4)), (), Span(0, 4))


if __name__ == '__main__': unittest.main()
