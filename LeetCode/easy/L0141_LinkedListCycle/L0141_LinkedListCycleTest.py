import unittest
import L0141_LinkedListCycle as leet
from LeetCode.common.ListNode import ListNode


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        head = [3, 2, 0, -4]
        pos = 1
        head = ListNode.build_list_node(head, pos)
        expect = True
        actual = _s.hasCycle(head)
        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        head = [1, 2]
        pos = 0
        head = ListNode.build_list_node(head, pos)
        expect = True
        actual = _s.hasCycle(head)
        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        head = [1]
        pos = -1
        head = ListNode.build_list_node(head, pos)
        expect = False
        actual = _s.hasCycle(head)
        self.assertEqual(expect, actual)

    def test_4(self):
        _s = leet.Solution()
        head = [1, 2]
        pos = -1
        head = ListNode.build_list_node(head, pos)
        expect = False
        actual = _s.hasCycle(head)
        self.assertEqual(expect, actual)

    def test_5(self):
        _s = leet.Solution()
        head = [1, 1,1,1]
        pos = -1
        head = ListNode.build_list_node(head, pos)
        expect = False
        actual = _s.hasCycle(head)
        self.assertEqual(expect, actual)

if __name__ == '__main__':
    unittest.main()
