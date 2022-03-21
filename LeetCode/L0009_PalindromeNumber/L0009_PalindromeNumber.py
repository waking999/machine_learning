class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False

        x_str = str(x)
        i = 0;
        x_str_len = len(x_str)
        x_str_half = x_str_len // 2
        while i <= x_str_half:
            if x_str[i] != x_str[x_str_len - i - 1]:
                return False
            i += 1

        return True
