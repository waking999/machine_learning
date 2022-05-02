import unittest
import L2220_MinimumBitFlipstoConvertNumber as leet

class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        start = 10
        goal = 7
        expect = 3
        actual = _s.minBitFlips(start, goal)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        start = 3
        goal = 4
        expect = 3
        actual = _s.minBitFlips(start, goal)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        start = 10
        goal = 82
        expect = 3
        actual = _s.minBitFlips(start, goal)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
