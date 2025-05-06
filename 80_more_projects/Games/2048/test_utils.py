# test_utils.py
import unittest
from utils import merge_row, transpose, boards_differ


class Test2048Utils(unittest.TestCase):

    def test_merge_row(self):
        self.assertEqual(merge_row([2, 2, 0, 0])[0], [4, 0, 0, 0])
        self.assertEqual(merge_row([2, 2, 4, 4])[0], [4, 8, 0, 0])
        self.assertEqual(merge_row([2, 0, 2, 2])[0], [4, 2, 0, 0])

    def test_transpose(self):
        self.assertEqual(transpose([[1, 2], [3, 4]]), [[1, 3], [2, 4]])

    def test_boards_differ(self):
        self.assertTrue(boards_differ([[1, 2], [3, 4]], [[1, 2], [3, 5]]))
        self.assertFalse(boards_differ([[1, 2], [3, 4]], [[1, 2], [3, 4]]))


if __name__ == "__main__":
    unittest.main()
