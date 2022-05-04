from typing import List


class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        _len = len(coins)
        dp = [0 for i in range(amount + 1)]

        for i in range(1, amount + 1):
            for j in range(_len):
                if coins[j] <= i:
                    dp[i] = min(dp[i], dp[i - coins[j]] + 1)

        return -1 if dp[amount] > amount else dp[amount]
