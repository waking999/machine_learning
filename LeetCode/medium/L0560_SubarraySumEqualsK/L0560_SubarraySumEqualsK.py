from typing import List


class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        d = {0: 1}
        count = 0
        _sum = 0
        for n in nums:
            _sum += n
            if _sum - k in d.keys():
                count += d[_sum - k]
            if _sum in d.keys():
                d[_sum] += 1
            else:
                d[_sum] = 1

        return count
