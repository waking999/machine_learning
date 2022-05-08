import unittest
import L0215_KthLargestElementinanArray as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        nums = [3, 2, 1, 5, 6, 4]
        k = 2
        expect = 5
        actual = _s.findKthLargest(nums, k)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        nums = [1]
        k = 1
        expect = 1
        actual = _s.findKthLargest(nums, k)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
