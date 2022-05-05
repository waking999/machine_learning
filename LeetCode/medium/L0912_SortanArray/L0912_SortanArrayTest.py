import unittest
import L0912_SortanArray as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        nums = [5, 2, 3, 1]
        expect = [1, 2, 3, 5]
        actual = _s.sortArray(nums)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
