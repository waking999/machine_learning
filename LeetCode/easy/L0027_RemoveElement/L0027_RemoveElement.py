from typing import List


class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        s = []
        for i in nums:
            if i != val:
                s.append(i)

        s_len = len(s)
        for i in range(s_len):
            nums[i]=s[i]

        return s_len
