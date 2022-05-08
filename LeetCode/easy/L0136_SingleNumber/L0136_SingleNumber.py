from typing import List


class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        _s = []
        for i in nums:
            if i in _s:
                _s.remove(i)
            else:
                _s.append(i)
        return _s[0]
