import unittest
import L0136_SingleNumber as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        nums = [2, 2, 1]
        expect = 1
        actual = _s.singleNumber(nums)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        nums = [4, 1, 2, 1, 2]
        expect = 4
        actual = _s.singleNumber(nums)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        nums = [1]
        expect = 1
        actual = _s.singleNumber(nums)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
