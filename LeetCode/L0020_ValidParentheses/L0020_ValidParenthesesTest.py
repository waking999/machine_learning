import unittest
import L0020_ValidParentheses as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        s = "()"
        expect = True
        actual = _s.isValid(s)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        s = "()[]{}"
        expect = True
        actual = _s.isValid(s)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        s = "(]"
        expect = False
        actual = _s.isValid(s)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
