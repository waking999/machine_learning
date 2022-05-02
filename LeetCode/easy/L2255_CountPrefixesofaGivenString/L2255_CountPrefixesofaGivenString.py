from typing import List


class Solution:
    def countPrefixes(self, words: List[str], s: str) -> int:
        count = 0
        l_s = len(s)
        for w in words:
            l_w = len(w)
            if l_w <= l_s and s[:l_w] == w:
                count += 1
        return count
