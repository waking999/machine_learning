from typing import List


class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:

        _len = len(nums)
        _sum = [0 for i in range(_len + 1)]

        for i in range(_len):
            _sum[i + 1] = _sum[i] + nums[i]

        d = {}
        for i in range(_len+1):
            m = _sum[i] % k
            if m in d.keys():
                if i - d[m] > 1:
                    return True
            else:
                d[m] = i

        return False
