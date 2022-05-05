from typing import List


class Solution:
    def sort(self, nums: List[int]) -> List[int]:
        n = len(nums)
        gap = n // 2
        while gap > 0:
            for i in range(gap, n):
                ins = nums[i]
                j = i - gap
                while j >= 0 and ins < nums[j]:
                    nums[j + gap] = nums[j]
                    j -= gap

                nums[j + gap] = ins

            gap //= 2

        return nums
