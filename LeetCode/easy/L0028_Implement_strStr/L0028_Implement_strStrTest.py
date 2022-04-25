import unittest
import L0028_Implement_strStr as leet

class L0028_Implement_strStrTest(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        haystack = "hello"
        needle = "ll"
        expect = 2
        actual = _s.strStr(haystack,needle)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        haystack = "aaaaa"
        needle = "bba"
        expect = -1
        actual = _s.strStr(haystack,needle)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        haystack = "mississippi"
        needle = "issip"
        expect = 4
        actual = _s.strStr(haystack, needle)
        self.assertEqual(expect, actual)



if __name__ == '__main__':
    unittest.main()
