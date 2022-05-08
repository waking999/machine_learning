from typing import List


class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        p1 = 0
        p2 = 1
        _l = len(nums)
        while p1 < _l and p2 < _l:
            if nums[p1] == nums[p2]:
                p2 += 1
            else:
                p1 += 1
                nums[p1] = nums[p2]
                p2 += 1

        return p1+1
