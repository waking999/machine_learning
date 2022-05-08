class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        p = -1
        len_s = len(s)
        while p >= -len_s and s[p] == ' ':
            p -= 1

        p1 = p

        while p >= -len_s and s[p] != ' ':
            p -= 1

        return p1 - p
