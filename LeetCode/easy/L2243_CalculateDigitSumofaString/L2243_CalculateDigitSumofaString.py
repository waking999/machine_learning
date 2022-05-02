class Solution:
    def digitSum(self, s: str, k: int) -> str:
        t_s = s
        l_s = len(t_s)

        while l_s > k:
            g = l_s // k
            tmp_s = ''
            for i in range(g):
                sum = 0
                for j in range(i * k, (i + 1) * k):
                    sum += int(t_s[j])
                tmp_s += str(sum)
            sum = 0
            for i in range(g * k, l_s):
                sum += int(t_s[i])
            tmp_s += str(sum)
            l_s = len(tmp_s)
            t_s = tmp_s

        return t_s
