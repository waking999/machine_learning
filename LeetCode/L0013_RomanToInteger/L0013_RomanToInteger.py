class Solution:
    def romanToInt(self, s: str) -> int:
        romanToIntMap = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000,'IV':4,'IX':9,'XL':40, 'XC':90,'CD':400,'CM':900 }
        # intToRomanMap = {1: 'I', 5: 'V', 10: 'X', 50: 'L', 100: 'C', 500: 'D', 1000: 'M'}

        i = 0
        sum = 0
        s_len = len(s)
        while i < s_len:
            if i + 1 < s_len and s[i:i + 2] in romanToIntMap:
                sum += romanToIntMap[s[i:i + 2]]
                i += 2
            else:
                sum += romanToIntMap[s[i]]
                i += 1
        return sum
