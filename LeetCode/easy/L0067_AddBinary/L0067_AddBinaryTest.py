import unittest
import L0067_AddBinary as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        a = "11"
        b = "1"
        expect = "100"
        actual = _s.addBinary(a, b)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        a = "1010"
        b = "1011"
        expect = "10101"
        actual = _s.addBinary(a, b)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        a = "1111"
        b = "1111"
        expect = "11110"
        actual = _s.addBinary(a, b)
        self.assertEqual(expect, actual)

if __name__ == '__main__':
    unittest.main()
