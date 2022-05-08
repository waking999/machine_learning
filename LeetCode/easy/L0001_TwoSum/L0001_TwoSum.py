from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        map = {}
        for ind, n in enumerate(nums):
            exp = target - n
            if exp in map:
                return [map[exp], ind]
            else:
                map[n] = ind
