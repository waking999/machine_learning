import unittest
import L0062_UniquePaths as leet

class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        m = 3
        n = 7
        expect = 28
        actual = _s.uniquePaths(m,n)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
