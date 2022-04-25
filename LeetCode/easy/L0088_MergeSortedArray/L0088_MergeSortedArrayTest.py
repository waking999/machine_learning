import unittest
import L0088_MergeSortedArray as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        nums1 = [1, 2, 3, 0, 0, 0]
        m = 3
        nums2 = [2, 5, 6]
        n = 3
        expect = [1, 2, 2, 3, 5, 6]
        _s.merge(nums1, m, nums2, n)

        self.assertEqual(expect, nums1)


def test_2(self):
    _s = leet.Solution()
    nums1 = [1]
    m = 1
    nums2 = []
    n = 0
    expect = [1]
    _s.merge(nums1, m, nums2, n)
    self.assertEqual(expect, nums1)


if __name__ == '__main__':
    unittest.main()
