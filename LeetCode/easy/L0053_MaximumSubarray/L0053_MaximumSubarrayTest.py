import unittest
import L0053_MaximumSubarray as leet


class L0053_MaximumSubarrayTest(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
        expect = 6
        actual = _s.maxSubArray(nums)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        nums = [1]
        expect = 1
        actual = _s.maxSubArray(nums)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        nums = [5, 4, -1, 7, 8]
        expect = 23
        actual = _s.maxSubArray(nums)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
