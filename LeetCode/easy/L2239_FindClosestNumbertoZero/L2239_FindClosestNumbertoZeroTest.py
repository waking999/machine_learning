import unittest
import L2239_FindClosestNumbertoZero as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        nums = [-4, -2, 1, 4, 8]
        expect = 1
        actual = _s.findClosestNumber(nums)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        nums = [-1000, -1000]
        expect = -1000
        actual = _s.findClosestNumber(nums)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        nums = [2,-1,1]
        expect = 1
        actual = _s.findClosestNumber(nums)
        self.assertEqual(expect, actual)

if __name__ == '__main__':
    unittest.main()
