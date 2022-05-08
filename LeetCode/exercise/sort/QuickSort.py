from typing import List


class Solution:
    def sort(self, nums: List[int]) -> List[int]:
        def partition(nums, left, right):
            i = left
            pivot = nums[right]
            for j in range(left, right):
                if nums[j] < pivot:
                    t = nums[i]
                    nums[i] = nums[j]
                    nums[j] = t
                    i += 1

            nums[right] = nums[i]
            nums[i] = pivot

            return i

        def quick_sort(nums, left, right):
            if left < right:
                center = partition(nums, left, right)
                quick_sort(nums, left, center - 1)
                quick_sort(nums, center + 1, right)

        quick_sort(nums, 0, len(nums)-1)
        return nums
