import unittest
import L0160_IntersectionofTwoLinkedLists as leet
from LeetCode.common.ListNode import ListNode


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        headA = [4, 1, 8, 4, 5]
        headA = ListNode.build_list_node(headA)
        headB = [5, 6, 1, 8, 4, 5]
        headB = ListNode.build_list_node(headB)
        expect = 8
        actual = _s.getIntersectionNode(headA, headB)
        self.assertEqual(expect, actual.val)


if __name__ == '__main__':
    unittest.main()
