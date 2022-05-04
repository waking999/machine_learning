import unittest
import L0072_EditDistance as leet

class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        word1 = "horse"
        word2 = "ros"
        expect = 3
        actual = _s.minDistance(word1,word2)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
