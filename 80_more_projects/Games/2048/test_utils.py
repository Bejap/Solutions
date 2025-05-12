# test_utils.py
import unittest
from utils import merge_row, transpose, boards_differ


class Test2048Utils(unittest.TestCase):

    def test_merge_row(self):
        row = [2, 2, 0, 0]
        merged, reward = merge_row(row)
        self.assertEqual(merged, [4, 0, 0, 0])
        self.assertEqual(reward, 4)  # 2+2=4

        row = [2, 2, 4, 4]
        merged, reward = merge_row(row)
        self.assertEqual(merged, [4, 8, 0, 0])
        self.assertEqual(reward, 12)  # 2+2=4 and 4+4=8

        row = [2, 0, 2, 2]
        merged, reward = merge_row(row)
        self.assertEqual(merged, [4, 2, 0, 0])
        self.assertEqual(reward, 4)  # 2+2=4 (last two)

        row = [4, 4, 4, 4]
        merged, reward = merge_row(row)
        self.assertEqual(merged, [8, 8, 0, 0])
        self.assertEqual(reward, 16)  # 4+4=8 and 4+4=8

        row = [2, 2, 2, 2]
        merged, reward = merge_row(row)
        self.assertEqual(merged, [4, 4, 0, 0])
        self.assertEqual(reward, 8)  # 2+2=4 and 2+2=4

        row = [0, 0, 0, 0]
        merged, reward = merge_row(row)
        self.assertEqual(merged, [0, 0, 0, 0])
        self.assertEqual(reward, 0)  # nothing to merge

        row = [4, 0, 0, 4]
        merged, reward = merge_row(row)
        self.assertEqual(merged, [8, 0, 0, 0])
        self.assertEqual(reward, 8)

    def test_transpose(self):
        self.assertEqual(transpose([[1, 2], [3, 4]]), [[1, 3], [2, 4]])

    def test_boards_differ(self):
        self.assertTrue(boards_differ([[1, 2], [3, 4]], [[1, 2], [3, 5]]))
        self.assertFalse(boards_differ([[1, 2], [3, 4]], [[1, 2], [3, 4]]))

