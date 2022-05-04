from typing import List


class Solution:
    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        _len = len(nums)
        _sum = [0 for i in range(_len + 1)]

        for i in range(_len):
            _sum[i + 1] = _sum[i] + nums[i]

        d = {}
        count=0
        for i in range(_len + 1):
            m = _sum[i] % k
            if m in d.keys():
                count+=d[m]
                d[m] += 1
            else:
                d[m] = 1

        # count = 0
        # for i in d.keys():
        #     if d[i]>=2:
        #         count += self.com(d[i], 2)

        return count

    def com(self, m, n):
        if n > m:
            return self.com(n, m)
        a = m - n
        a1 = min(n, a)
        a2 = max(n, a)
        p1 = 1
        for i in range(1, a1 + 1):
            p1 *= i

        p2 = p1
        for i in range(a1 + 1, a2 + 1):
            p2 *= i

        p3 = p2
        for i in range(a2 + 1, m + 1):
            p3 *= i

        return p3 // p2 // p1
