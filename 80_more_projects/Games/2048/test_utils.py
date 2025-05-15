# test_utils.py
import unittest
from utils import merge_row, transpose, boards_differ


class Test2048Utils(unittest.TestCase):

    def test_merge_row(self):
        row = [2, 2, 0, 0]
        merged = merge_row(row)
        self.assertEqual(merged, [4, 0, 0, 0])

        row = [2, 2, 4, 4]
        merged = merge_row(row)
        self.assertEqual(merged, [4, 8, 0, 0])

        row = [2, 0, 2, 2]
        merged = merge_row(row)
        self.assertEqual(merged, [4, 2, 0, 0])

        row = [4, 4, 4, 4]
        merged = merge_row(row)
        self.assertEqual(merged, [8, 8, 0, 0])

        row = [2, 2, 2, 2]
        merged = merge_row(row)
        self.assertEqual(merged, [4, 4, 0, 0])

        row = [0, 0, 0, 0]
        merged = merge_row(row)
        self.assertEqual(merged, [0, 0, 0, 0])

        row = [4, 0, 0, 4]
        merged = merge_row(row)
        self.assertEqual(merged, [8, 0, 0, 0])

    def test_transpose(self):
        self.assertEqual(transpose([[1, 2], [3, 4]]), [[1, 3], [2, 4]])

    def test_boards_differ(self):
        self.assertTrue(boards_differ([[1, 2], [3, 4]], [[1, 2], [3, 5]]))
        self.assertFalse(boards_differ([[1, 2], [3, 4]], [[1, 2], [3, 4]]))

