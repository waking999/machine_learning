import unittest
import L0560_SubarraySumEqualsK as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        nums = [1, 1, 1]
        k = 2
        expect = 2
        actual = _s.subarraySum(nums, k)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        nums = [1, 2, 3]
        k = 3
        expect = 2
        actual = _s.subarraySum(nums, k)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
