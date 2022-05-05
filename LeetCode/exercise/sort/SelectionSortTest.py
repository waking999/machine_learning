import unittest

import SelectionSort as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        nums = [4, 1, 3, 2, 7, 5, 8, 0]
        expect = [0, 1, 2, 3, 4, 5, 7, 8]
        actual = _s.sort(nums)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
