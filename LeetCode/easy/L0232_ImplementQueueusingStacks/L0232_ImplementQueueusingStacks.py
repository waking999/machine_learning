class MyQueue:

    def __init__(self):
        self._list = []
        self.head = 0
        self.tail = 0

    def push(self, x: int) -> None:
        if self.tail >= len(self._list):
            self._list.append(x)
        else:
            self._list[self.tail] = x

        self.tail += 1

    def pop(self) -> int:
        x = self._list[self.head]
        self.head += 1
        return x

    def peek(self) -> int:
        x = self._list[self.head]
        return x

    def empty(self) -> bool:
        return self.tail - self.head == 0
