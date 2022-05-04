from typing import List


class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0

        dp = [0 for i in range(n)]
        dp[0] = nums[0]
        if n < 2:
            return nums[0];
        dp[1] = nums[0] if nums[0] > nums[1] else nums[1]

        for i in range(2, n):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])

        return dp[n-1]
