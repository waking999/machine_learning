import unittest
from LeetCode.L0001_TwoSum import L0001_TwoSum as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        s = leet.Solution()
        nums = [2, 7, 11, 15]
        target = 9
        expect = [0, 1]
        actual = s.twoSum(nums, target)
        self.assertEqual(expect, actual)

    def test_2(self):
        s = leet.Solution()
        nums = [3, 2, 4]
        target = 6
        expect = [1, 2]
        actual = s.twoSum(nums, target)
        self.assertEqual(expect, actual)

    def test_3(self):
        s = leet.Solution()
        nums = [3, 3]
        target = 6
        expect = [0, 1]
        actual = s.twoSum(nums, target)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
