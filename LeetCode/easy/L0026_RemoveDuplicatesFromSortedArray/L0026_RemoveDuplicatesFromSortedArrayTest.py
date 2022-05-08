import unittest
import L0026_RemoveDuplicatesFromSortedArray as leet


class L0026_RemoveDuplicatesFromSortedArrayTest(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        nums = [1, 1, 2]
        expect = 2
        actual = _s.removeDuplicates(nums)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
        expect = 5
        actual = _s.removeDuplicates(nums)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
