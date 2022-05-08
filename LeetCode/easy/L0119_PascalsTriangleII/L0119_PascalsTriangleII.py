import math
from typing import List



class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        rtn_list = []
        for i in range(rowIndex+1):
            row_list = [1 for j in range(i + 1)]
            for j in range(1, math.ceil((1 + i) / 2)):
                t = rtn_list[i - 1][j - 1] + rtn_list[i - 1][j]
                row_list[j] = t
                row_list[i - j] = t
            rtn_list.append(row_list)

        return rtn_list[rowIndex]
