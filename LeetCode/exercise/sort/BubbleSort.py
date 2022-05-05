from typing import List


class Solution:
    def sort(self, nums: List[int]) -> List[int]:
        n = len(nums)
        for i in range(n):
            for j in range(n - i - 1):
                if nums[j] > nums[j + 1]:
                    t = nums[j]
                    nums[j] = nums[j + 1]
                    nums[j + 1] = t

        return nums
