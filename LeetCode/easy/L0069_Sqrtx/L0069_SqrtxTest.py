import unittest
import L0069_Sqrtx as leet

class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        x = 4
        expect = 2
        actual = _s.mySqrt(x)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        x = 8
        expect = 2
        actual = _s.mySqrt(x)
        self.assertEqual(expect, actual)



    def test_3(self):
        _s = leet.Solution()
        x = 1
        expect = 1
        actual = _s.mySqrt(x)
        self.assertEqual(expect, actual)


    def test_4(self):
        _s = leet.Solution()
        x = 5
        expect = 2
        actual = _s.mySqrt(x)
        self.assertEqual(expect, actual)

if __name__ == '__main__':
    unittest.main()
