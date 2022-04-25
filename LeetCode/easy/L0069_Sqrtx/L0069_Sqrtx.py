class Solution:
    def mySqrt(self, x: int) -> int:
        if x == 0:
            return 0
        if x < 4:
            return 1
        if x < 9:
            return 2

        start = 0
        end = x
        e = 1e-05
        while start < end:
            t = start + (end - start) / 2
            t2 = t ** 2
            if abs(t2 - x) <= e:
                return int(t)
            elif t2 > x:
                end = t
            else:
                start = t
        rtn = int(start)
        return rtn
