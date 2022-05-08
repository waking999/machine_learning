from typing import List


class Solution:
    def sort(self, nums: List[int]) -> List[int]:
        n = len(nums)
        for i in range(1, n):
            ins = nums[i]
            j = i - 1
            while j >= 0 and nums[j] > ins:
                nums[j + 1] = nums[j]
                j -= 1

            nums[j + 1] = ins

        return nums
