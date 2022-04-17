from typing import List
import sys

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        loc_sum = 0
        max_val = -99999
        for i in nums:
            if loc_sum < 0:
                loc_sum = i
            else:
                loc_sum = loc_sum + i

            if max_val < loc_sum:
                max_val = loc_sum

        return max_val
