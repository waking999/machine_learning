from typing import List


class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        p = 0
        _l = len(nums)
        while p < _l:
            if nums[p] < target:
                p += 1
            else:
                return p

        return p
