import unittest
import L0010_RegularExpressionMatching as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        s = "aa"
        p = "a"
        expect = False
        actual = _s.isMatch(s, p)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        s = "aa"
        p = "a*"
        expect = True
        actual = _s.isMatch(s, p)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        s = "ab"
        p = "*."
        expect = True
        actual = _s.isMatch(s, p)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
