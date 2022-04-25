import unittest
import L0083_RemoveDuplicatesfromSortedList as leet
from LeetCode.common.ListNode import ListNode


class MyTestCase(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        list = [1, 1, 2]
        head = ListNode.build_list_node(list)

        list3 = [1, 2]
        expect = ListNode.build_list_node(list3)
        actual = _s.deleteDuplicates(head)

        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        list = [1, 1, 2, 3, 3]
        head = ListNode.build_list_node(list)

        list3 = [1, 2, 3]
        expect = ListNode.build_list_node(list3)
        actual = _s.deleteDuplicates(head)

        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
