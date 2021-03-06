class Solution:
    def climbStairs(self, n: int) -> int:
        if n < 4:
            return n

        dp0 = 1
        dp1 = 1
        for i in range(2, n+1):
            dp1 = dp0 + dp1
            dp0 = dp1-dp0

        return dp1
