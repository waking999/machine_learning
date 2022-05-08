import unittest
import L2255_CountPrefixesofaGivenString as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        words = ["a", "b", "c", "ab", "bc", "abc"]
        s = "abc"
        expect = 3
        actual = _s.countPrefixes(words, s)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
