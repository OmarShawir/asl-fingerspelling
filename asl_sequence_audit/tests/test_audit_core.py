import unittest

from audit_core import alphabet_scope, ctc_minimum, hand_blocks, split_name


class AuditCoreTests(unittest.TestCase):
    def test_ctc_adjacent_repeats(self):
        self.assertEqual(ctc_minimum("hello"), 6)
        self.assertEqual(ctc_minimum("emmett"), 8)
        self.assertEqual(ctc_minimum("all"), 4)

    def test_ctc_repeated_run_and_nonadjacent(self):
        self.assertEqual(ctc_minimum("aaa"), 5)
        self.assertEqual(ctc_minimum("aba"), 3)
        self.assertEqual(ctc_minimum(""), 0)

    def test_scope_preserves_whole_labels(self):
        self.assertTrue(alphabet_scope("jazz club"))
        self.assertFalse(alphabet_scope("dr. smith"))
        self.assertFalse(alphabet_scope("a@b"))
        self.assertFalse(alphabet_scope(""))

    def test_source_partition_mapping(self):
        self.assertEqual(split_name("dev"), "validation")
        self.assertEqual(split_name("train"), "train")
        self.assertEqual(split_name("test"), "test")

    def test_one_hand_has_one_block(self):
        a = [[float(i), 0., 0.] for i in range(21)]
        self.assertEqual(hand_blocks(a), [a])
        self.assertEqual(hand_blocks([]), [])

    def test_two_hands_keep_separate_blocks(self):
        a = [[float(i), 0., 0.] for i in range(21)]
        b = [[float(i), 1., 0.] for i in range(21)]
        self.assertEqual(hand_blocks(a + b), [a, b])
        self.assertEqual(hand_blocks(b + a), [b, a])

    def test_invalid_shape_is_not_silently_padded(self):
        for frame in (None, [[0., 0., 0.]] * 20, [[0., 0.]] * 21):
            with self.assertRaises(ValueError):
                hand_blocks(frame)


if __name__ == "__main__":
    unittest.main()
