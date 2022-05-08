from typing import List


class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        p1 = 0
        p2 = 0
        t=m+n-1
        while p1 < m and p2 < n:
            if nums1[p1] > nums2[p2]:
                for i in range(t, p1, -1):
                    nums1[i] = nums1[i - 1]
                nums1[p1] = nums2[p2]
                p2 += 1
                m+=1
            p1 += 1

        while p2 < n:
            nums1[p1] = nums2[p2]
            p1 += 1
            p2 += 1
