import unittest
import L0027_RemoveElement as leet


class L0027_RemoveElementTest(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        nums = [3, 2, 2, 3]
        val = 3
        expect = 2
        actual = _s.removeElement(nums, val)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        nums = [0,1,2,2,3,0,4,2]
        val = 2
        expect = 5
        actual = _s.removeElement(nums, val)
        self.assertEqual(expect, actual)

if __name__ == '__main__':
    unittest.main()
