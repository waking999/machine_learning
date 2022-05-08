from typing import List


class Solution:
    def intersection(self, nums: List[List[int]]) -> List[int]:
        d = {}
        l_n = len(nums)

        for r in nums:
            for i in r:
                if i in d:
                    d[i] += 1
                else:
                    d[i] = 1

        rtn = []
        for k in d.keys():
            if d[k] == l_n:
                rtn.append(k)

        rtn.sort()
        return rtn
