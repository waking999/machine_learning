import unittest
import L0238_ProductofArrayExceptSelf as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        nums = [1, 2, 3, 4]
        expect = [24, 12, 8, 6]
        actual = _s.productExceptSelf(nums)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
