import unittest
import L0155_MinStack as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        minStack = leet.MinStack()
        minStack.push(-2)
        minStack.push(0)
        minStack.push(-3)
        a1 = minStack.getMin()  # return -3
        e1 = -3
        self.assertEqual(e1, a1)
        minStack.pop()
        a2 = minStack.top()  # return 0
        e2 = 0
        self.assertEqual(e2, a2)
        a3 = minStack.getMin()  # return -2
        e3 = -2
        self.assertEqual(e2, a2)

    def test_2(self):
        minStack = leet.MinStack()
        minStack.push(2147483646)
        minStack.push(2147483646)
        minStack.push(2147483647)
        a1 = minStack.top()
        e1 = 2147483647
        self.assertEqual(e1, a1)
        minStack.pop()
        a2 = minStack.getMin()
        e2 = 2147483646
        self.assertEqual(e2, a2)
        minStack.pop()
        a3 = minStack.getMin()
        e3 = 2147483646
        self.assertEqual(e3, a3)
        minStack.pop()
        minStack.push(2147483647)
        a4 = minStack.top()
        e4 = 2147483647
        self.assertEqual(e1, a1)
        a5 = minStack.getMin()
        e5 = 2147483647
        self.assertEqual(e5, a5)
        minStack.push(-2147483648)
        a6 = minStack.top()
        e6 = -2147483648
        self.assertEqual(e6, a6)
        a7 = minStack.getMin()
        e7 = -2147483648
        self.assertEqual(e7, a7)
        minStack.pop()
        a8 = minStack.getMin()
        e8 = 2147483647
        self.assertEqual(e8, a8)

    def test_3(self):
        minStack = leet.MinStack()
        minStack.push(-10)
        minStack.push(14)
        a1 = minStack.getMin()
        e1 = -10
        self.assertEqual(e1, a1)
        a2 = minStack.getMin()
        e2 = -10
        self.assertEqual(e2, a2)
        minStack.push(-20)
        a3 = minStack.getMin()
        e3 = -20
        self.assertEqual(e3, a3)
        a4 = minStack.getMin()
        e4 = -20
        self.assertEqual(e4, a4)
        a5 = minStack.top()
        e5 = -20
        self.assertEqual(e5, a5)
        a6 = minStack.getMin()
        e6 = -20
        self.assertEqual(e6, a6)
        minStack.pop()
        minStack.push(10)
        minStack.push(-7)
        a7 = minStack.getMin()
        e7 = -10
        self.assertEqual(e7, a7)
        minStack.push(-7)
        minStack.pop()
        a8 = minStack.top()
        e8 = -7
        self.assertEqual(e8, a8)
        a9 = minStack.getMin()
        e9 = -10
        self.assertEqual(e9, a9)
        minStack.pop()

    def test_4(self):
        minStack = leet.MinStack()
        minStack.push(2)
        minStack.push(0)
        minStack.push(3)
        minStack.push(0)
        a1 = minStack.getMin()
        e1 = 0
        self.assertEqual(e1, a1)
        minStack.pop()
        a2 = minStack.getMin()
        e2 = 0
        self.assertEqual(e2, a2)
        minStack.pop()
        a3 = minStack.getMin()
        e3 = 0
        self.assertEqual(e3, a3)
        minStack.pop()
        a4 = minStack.getMin()
        e4 = 2
        self.assertEqual(e4, a4)


if __name__ == '__main__':
    unittest.main()
