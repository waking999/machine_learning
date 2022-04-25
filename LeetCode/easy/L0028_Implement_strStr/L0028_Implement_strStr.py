class Solution:
    def strStr(self, haystack: str, needle: str) -> int:

        len_h = len(haystack)
        len_n = len(needle)
        if len_h < len_n:
            return -1
        if len_n == len_h and haystack == needle:
            return 0

        p_h = 0
        p_n = 0
        while p_h < len_h and p_n < len_n and len_h - p_h >= len_n - p_n:
            if haystack[p_h] != needle[p_n]:
                p_h += 1
            else:
                k = p_h
                while p_h < len_h and p_n < len_n and len_h - p_h >= len_n - p_n and haystack[p_h] == needle[p_n]:
                    p_h += 1
                    p_n += 1

                if p_n == len_n:
                    return k
                else:
                    p_h = k + 1
                    p_n = 0

        return -1
