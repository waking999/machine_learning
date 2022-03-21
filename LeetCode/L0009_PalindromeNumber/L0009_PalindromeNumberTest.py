import unittest
import L0009_PalindromeNumber as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        s = leet.Solution()
        x = 121
        expect = True
        actual = s.isPalindrome(x)
        self.assertEqual(expect, actual)


    def test_2(self):
        s = leet.Solution()
        x = -121
        expect = False
        actual = s.isPalindrome(x)
        self.assertEqual(expect, actual)


    def test_3(self):
        s = leet.Solution()
        x = 10
        expect = False
        actual = s.isPalindrome(x)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
