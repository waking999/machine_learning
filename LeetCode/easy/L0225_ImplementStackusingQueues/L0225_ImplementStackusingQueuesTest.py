import unittest
import L0225_ImplementStackusingQueues as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        myStack = leet.MyStack()
        myStack.push(1)
        myStack.push(2)
        e1 = 2
        a1 = myStack.top()
        self.assertEqual(e1, a1)
        e2 = 2
        a2 = myStack.pop()
        self.assertEqual(e2, a2)
        e3 = False
        a3 = myStack.empty()
        self.assertEqual(e3,a3)
        return


if __name__ == '__main__':
    unittest.main()
