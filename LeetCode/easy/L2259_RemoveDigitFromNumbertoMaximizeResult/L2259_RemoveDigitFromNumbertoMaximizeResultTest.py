import unittest
import L2259_RemoveDigitFromNumbertoMaximizeResult as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        number = "123"
        digit = "3"
        expect = "12"
        actual = _s.removeDigit(number, digit)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        number = "1231"
        digit = "1"
        expect = "231"
        actual = _s.removeDigit(number, digit)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        number = "551"
        digit = "5"
        expect = "51"
        actual = _s.removeDigit(number, digit)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
