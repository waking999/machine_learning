class Solution:
    def removeDigit(self, number: str, digit: str) -> str:
        _max = -1
        for i in range(len(number)):
            if number[i] == digit:
                _max = max(_max, int(number[:i] + number[i + 1:]))

        return str(_max)
