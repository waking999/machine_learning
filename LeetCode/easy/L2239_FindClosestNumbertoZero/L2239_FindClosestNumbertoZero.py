import sys
from typing import List


class Solution:
    def findClosestNumber(self, nums: List[int]) -> int:
        _min = sys.maxsize

        for n in nums:
            _t = abs(n)
            if _t < _min:
                _min = _t

        _rtn = []
        for n in nums:
            _t = abs(n)
            if _t == _min:
                _rtn.append(n)

        _rtn.sort()
        return _rtn[-1]
