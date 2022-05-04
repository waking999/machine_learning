import unittest
import L0198_HouseRobber as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        nums = [1, 2, 3, 1]
        expect = 4
        actual = _s.rob(nums)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        nums = [2, 7, 9, 3, 1]
        expect = 12
        actual = _s.rob(nums)
        self.assertEqual(expect, actual)

if __name__ == '__main__':
    unittest.main()
