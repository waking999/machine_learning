import unittest
import L2215_FindtheDifferenceofTwoArrays as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        nums1 = [1, 2, 3]
        nums2 = [2, 4, 6]
        expect = [[1, 3], [4, 6]]
        actual = _s.findDifference(nums1, nums2)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        nums1 = [1, 2, 3, 3]
        nums2 = [1, 1, 2, 2]
        expect = [[3], []]
        actual = _s.findDifference(nums1, nums2)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
