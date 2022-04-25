import unittest
import L0035_SearchInsertPosition as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        nums = [1, 3, 5, 6]
        target = 5
        expect = 2
        actual = _s.searchInsert(nums, target)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        nums = [1, 3, 5, 6]
        target = 2
        expect = 1
        actual = _s.searchInsert(nums, target)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        nums = [1, 3, 5, 6]
        target = 7
        expect = 4
        actual = _s.searchInsert(nums, target)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
