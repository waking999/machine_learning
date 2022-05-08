import unittest
import L0125_ValidPalindrome as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        s = "A man, a plan, a canal: Panama"
        expect = True
        actual = _s.isPalindrome(s)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        s = "race a car"
        expect = False
        actual = _s.isPalindrome(s)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        s = " "
        expect = True
        actual = _s.isPalindrome(s)
        self.assertEqual(expect, actual)

    def test_4(self):
        _s = leet.Solution()
        s = ".,"
        expect = True
        actual = _s.isPalindrome(s)
        self.assertEqual(expect, actual)

    def test_5(self):
        _s = leet.Solution()
        s = "0P"
        expect = False
        actual = _s.isPalindrome(s)
        self.assertEqual(expect, actual)

    def test_6(self):
        _s = leet.Solution()
        s = "ab2a"
        expect = False
        actual = _s.isPalindrome(s)
        self.assertEqual(expect, actual)

if __name__ == '__main__':
    unittest.main()
