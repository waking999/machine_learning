import unittest
import L0070_ClimbingStairs as leet

class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        n = 2
        expect = 2
        actual = _s.climbStairs(n)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        n = 3
        expect = 3
        actual = _s.climbStairs(n)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        n = 4
        expect = 5
        actual = _s.climbStairs(n)
        self.assertEqual(expect, actual)

if __name__ == '__main__':
    unittest.main()
