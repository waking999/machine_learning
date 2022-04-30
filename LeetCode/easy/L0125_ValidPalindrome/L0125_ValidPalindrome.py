class Solution:
    def isPalindrome(self, s: str) -> bool:
        _l = len(s)
        p1 = 0
        p2 = _l - 1
        while p1 < p2:
            while not (('A' <= s[p1] <= 'Z') or ('a' <= s[p1] <= 'z') or ('0' <= s[p1] <= '9')):
                p1 += 1
                if p1 >= _l:
                    return True

            while not (('A' <= s[p2] <= 'Z') or ('a' <= s[p2] <= 'z') or ('0' <= s[p2] <= '9')):
                p2 -= 1
                if p2 < 0:
                    return True

            sp1_a = ord(s[p1])
            sp2_a = ord(s[p2])
            if sp1_a == sp2_a or (sp1_a >= 65 and sp2_a >= 65 and abs(sp1_a - sp2_a) == 32):
                p1 += 1
                p2 -= 1
            else:
                return False

        return True
