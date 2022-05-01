import sys


class MinStack:

    def __init__(self):
        self._list = []
        self._point = -1

        self._min_list = []
        self._min_list_point = -1

    def push(self, val: int) -> None:
        self._point += 1
        if self._point < len(self._list):
            self._list[self._point] = val
        else:
            self._list.append(val)

        if self._min_list_point < 0 or val <= self._min_list[self._min_list_point]:
            self._min_list_point += 1
            if self._min_list_point < len(self._min_list):
                self._min_list[self._min_list_point] = val
            else:
                self._min_list.append(val)

    def pop(self) -> None:
        if self._list[self._point] == self._min_list[self._min_list_point]:
            self._min_list_point -= 1

        self._point -= 1

    def top(self) -> int:
        return self._list[self._point]

    def getMin(self) -> int:
        return self._min_list[self._min_list_point]
