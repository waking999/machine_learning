from typing import List
import sys


class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        dp = [0 for i in range(n)]
        dp[0] = nums[0]
        _max = dp[0]
        for i in range(1,n):
            dp[i] = max(dp[i - 1] + nums[i], nums[i])
            if _max < dp[i]:
                _max = dp[i]

        return _max

        # loc_sum = 0
        # max_val = -99999
        # for i in nums:
        #     if loc_sum < 0:
        #         loc_sum = i
        #     else:
        #         loc_sum = loc_sum + i
        #
        #     if max_val < loc_sum:
        #         max_val = loc_sum
        #
        # return max_val
