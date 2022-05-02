import unittest
import L2248_IntersectionofMultipleArrays as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        nums = [[3, 1, 2, 4, 5], [1, 2, 3, 4], [3, 4, 5, 6]]
        expect = [3, 4]
        actual = _s.intersection(nums)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
