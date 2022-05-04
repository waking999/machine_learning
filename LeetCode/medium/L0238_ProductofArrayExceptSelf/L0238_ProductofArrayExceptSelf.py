from typing import List


class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        _len = len(nums)

        fwd = [1 for i in range(_len)]
        bwd = [1 for i in range(_len)]

        for i in range(_len - 1):
            fwd[i + 1] = fwd[i] * nums[i]

        for i in range(_len - 1, 0, -1):
            bwd[i - 1] = bwd[i] * nums[i]

        rtn=[]
        for i in range(_len):
            rtn.append(fwd[i] * bwd[i])

        return rtn
