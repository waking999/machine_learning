import unittest
import L0058_LengthofLastWord as leet


class L0058_LengthofLastWordTest(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        s = "Hello World"
        expect = 5
        actual = _s.lengthOfLastWord(s)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        s = "   fly me   to   the moon  "
        expect = 4
        actual = _s.lengthOfLastWord(s)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        s = "luffy is still joyboy"
        expect = 6
        actual = _s.lengthOfLastWord(s)
        self.assertEqual(expect, actual)

    def test_4(self):
        _s = leet.Solution()
        s = "a"
        expect = 1
        actual = _s.lengthOfLastWord(s)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
