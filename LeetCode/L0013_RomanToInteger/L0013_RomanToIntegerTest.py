import unittest
import L0013_RomanToInteger as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        s = 'III'
        expect = 3
        actual = _s.romanToInt(s)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        s = 'MCMXCIV'
        expect = 1994
        actual = _s.romanToInt(s)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        s = 'LVIII'
        expect = 58
        actual = _s.romanToInt(s)
        self.assertEqual(expect, actual)

if __name__ == '__main__':
    unittest.main()
