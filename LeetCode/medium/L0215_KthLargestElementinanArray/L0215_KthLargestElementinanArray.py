from typing import List


class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        find_flag = False
        find_num = None

        n = len(nums)

        def partition(nums, left, right):

            pivot = nums[left]
            l = left + 1
            r = right
            while l <= r:
                if nums[l] < pivot and nums[r] > pivot:
                    t = nums[l]
                    nums[l] = nums[r]
                    nums[r] = t
                if nums[l] >= pivot:
                    l += 1
                if nums[r] <= pivot:
                    r -= 1
            nums[left] = nums[r]
            nums[r] = pivot
            return r

        def find(nums, left, right):
            nonlocal find_num
            nonlocal find_flag

            if find_flag:
                return

            if left <= right:
                pos = partition(nums, left, right)
                if pos == k - 1:
                    find_flag = True
                    find_num = nums[pos]
                    return
                elif pos > k - 1:
                    find(nums, left, pos - 1)
                else:
                    find(nums, pos + 1, right)

        find(nums, 0, n - 1)

        return find_num
