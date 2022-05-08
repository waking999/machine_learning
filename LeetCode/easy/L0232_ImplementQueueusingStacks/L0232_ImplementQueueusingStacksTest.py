import unittest
import L0232_ImplementQueueusingStacks as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        myQueue = leet.MyQueue()
        myQueue.push(1)
        myQueue.push(2)
        e1 = 1
        a1 = myQueue.peek()
        self.assertEqual(e1, a1)
        e2 = 1
        a2 = myQueue.pop()
        self.assertEqual(e2, a2)
        e3 = False
        a3 = myQueue.empty()
        self.assertEqual(e3, a3)
        return


if __name__ == '__main__':
    unittest.main()
