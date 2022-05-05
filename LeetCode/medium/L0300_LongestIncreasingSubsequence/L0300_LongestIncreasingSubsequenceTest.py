import unittest
import L0300_LongestIncreasingSubsequence as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        nums = [10, 9, 2, 5, 3, 7, 101, 18]
        expect = 4
        actual = _s.lengthOfLIS(nums)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
