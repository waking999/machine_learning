import unittest
import L0014_LongestCommonPrefix as leet


class L0020_ValidParenthesesTest(unittest.TestCase):
    def test_1(self):
        s = leet.Solution()
        strs = ["flower", "flow", "flight"]
        expect = "fl"
        actual = s.longestCommonPrefix(strs)
        self.assertEqual(expect, actual)

    def test_2(self):
        s = leet.Solution()
        strs = ["dog", "racecar", "car"]
        expect = ""
        actual = s.longestCommonPrefix(strs)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
