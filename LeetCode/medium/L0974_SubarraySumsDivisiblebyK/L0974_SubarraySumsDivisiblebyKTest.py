import unittest

import L0974_SubarraySumsDivisiblebyK as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        nums = [4, 5, 0, -2, -3, 1]
        k = 5
        expect = 7
        actual = _s.subarraysDivByK(nums, k)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
