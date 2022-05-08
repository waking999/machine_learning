import unittest
import L0322_CoinChange as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        coins = [1, 2, 5]
        amount = 11
        expect = 3
        actual = _s.coinChange(coins, amount)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        coins = coins = [2]
        amount = 3
        expect = -1
        actual = _s.coinChange(coins, amount)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        coins = [1]
        amount = 0
        expect = 0
        actual = _s.coinChange(coins, amount)
        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
