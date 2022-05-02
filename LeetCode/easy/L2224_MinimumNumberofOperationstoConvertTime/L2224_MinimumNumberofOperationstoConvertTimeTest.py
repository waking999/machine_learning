import unittest
import L2224_MinimumNumberofOperationstoConvertTime as leet

class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        current = "02:30"
        correct = "04:35"
        expect = 3
        actual = _s.convertTime(current, correct)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        current = "11:00"
        correct = "11:01"
        expect = 1
        actual = _s.convertTime(current, correct)
        self.assertEqual(expect, actual)





if __name__ == '__main__':
    unittest.main()
