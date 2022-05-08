from typing import List


class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        binary=50000
        n = len(nums)
        rtn = [None for i in range(n)]
        count = [0 for i in range(binary*2+1)]
        for num in nums:
            count[num + binary] += 1

        ind = 0
        for i in range(binary*2+1):
            c = count[i]
            while c > 0:
                rtn[ind] = i - binary
                c -= 1
                ind += 1

        return rtn
