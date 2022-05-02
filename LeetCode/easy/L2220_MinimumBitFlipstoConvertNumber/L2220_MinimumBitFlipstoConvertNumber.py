class Solution:
    def minBitFlips(self, start: int, goal: int) -> int:
        start_bin, start_len = self.int_to_bin(start)
        goal_bin, goal_len = self.int_to_bin(goal)
        count = 0
        for i in range(min(goal_len, start_len)):
            if goal_bin[i] != start_bin[i]:
                count += 1

        if start_len > goal_len:
            for i in range(goal_len, start_len):
                if start_bin[i] == 1:
                    count += 1
        else:
            for i in range(start_len, goal_len):
                if goal_bin[i] == 1:
                    count += 1

        return count

    def int_to_bin(self, num):
        rtn = []
        len = 0
        while num > 0:
            rtn.append(num % 2)
            num = num // 2
            len += 1

        return rtn, len
