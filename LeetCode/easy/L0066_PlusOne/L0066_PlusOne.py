from typing import List


class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        carrier = 1

        digits.reverse()
        for i in range(0, len(digits)):
            t = digits[i] + carrier
            if t == 10:
                digits[i] = 0
                carrier = 1
            else:
                digits[i] = t
                carrier = 0
        if carrier == 1:
            digits.append(1)

        digits.reverse()

        return digits
