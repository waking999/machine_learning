class Solution:
    def convertTime(self, current: str, correct: str) -> int:
        cur_hh = int(current[0:2])
        cur_mm = int(current[3:])

        cor_hh = int(correct[0:2])
        cor_mm = int(correct[3:])

        count = 0
        if cor_mm >= cur_mm:
            count += cor_hh - cur_hh
            minut = cor_mm - cur_mm
            count += minut // 15 + (minut % 15) // 5 + (minut % 15) % 5
        else:
            count += cor_hh - cur_hh - 1
            minut = cor_mm - cur_mm + 60
            count += minut // 15 + (minut % 15) // 5 + (minut % 15) % 5
        return count
