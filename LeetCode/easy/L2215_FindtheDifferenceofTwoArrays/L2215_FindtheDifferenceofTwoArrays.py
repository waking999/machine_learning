from typing import List


class Solution:
    def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
        nums1 = set(nums1)
        nums2 = set(nums2)

        nums3 = nums1.difference(nums2)
        nums4 = nums2.difference(nums1)

        return [list(nums3),list(nums4)]
