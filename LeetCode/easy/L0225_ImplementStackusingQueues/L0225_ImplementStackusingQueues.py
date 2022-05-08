class MyStack:

    def __init__(self):
        self.queue = []
        self.tail = 0

    def push(self, x: int) -> None:
        if self.tail >= len(self.queue):
            self.queue.append(x)
        else:
            self.queue[self.tail] = x

        self.tail += 1

    def pop(self) -> int:
        self.tail -= 1
        x = self.queue[self.tail]
        return x

    def top(self) -> int:
        return self.queue[self.tail - 1]

    def empty(self) -> bool:
        return self.tail == 0
