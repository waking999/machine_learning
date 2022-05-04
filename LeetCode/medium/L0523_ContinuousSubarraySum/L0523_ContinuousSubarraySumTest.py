import unittest
import L0523_ContinuousSubarraySum as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        nums = [23, 2, 4, 6, 7]
        k = 6
        expect = True
        actual = _s.checkSubarraySum(nums, k)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        nums = [23, 2, 6, 4, 7]
        k = 6
        expect = True
        actual = _s.checkSubarraySum(nums, k)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        nums = [23, 2, 6, 4, 7]
        k = 13
        expect = False
        actual = _s.checkSubarraySum(nums, k)
        self.assertEqual(expect, actual)

    def test_4(self):
        _s = leet.Solution()
        nums = [23, 2, 4, 6, 6]
        k = 7
        expect = True
        actual = _s.checkSubarraySum(nums, k)
        self.assertEqual(expect, actual)

    def test_5(self):
        _s = leet.Solution()
        nums = [0,0]
        k = 1
        expect = True
        actual = _s.checkSubarraySum(nums, k)
        self.assertEqual(expect, actual)

if __name__ == '__main__':
    unittest.main()
