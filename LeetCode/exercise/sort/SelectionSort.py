from typing import List


class Solution:
    def sort(self, nums: List[int]) -> List[int]:
        n = len(nums)
        for i in range(n):
            _mi = i
            for j in range(i + 1, n):
                if nums[j] < nums[_mi]:
                    _mi = j

            t = nums[i]
            nums[i] = nums[_mi]
            nums[_mi] = t

        return nums
