from typing import List


class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        pre = strs[0]
        for i in strs:
            pre=self.compare(pre, i)

        return pre

    def compare(self, a, b):
        min_len = min(len(a),len(b))
        pre = ""
        for i in range(min_len):
            if a[i]!=b[i]:
                break
            else:
                pre+=a[i]
        return pre