import unittest
import L0021_MergeTwoSortedLists as leet
from LeetCode.common.ListNode import ListNode


class L0021_MergeTwoSortedListsTest(unittest.TestCase):
    def test_1(self):
        _s = leet.Solution()
        list1 = [1, 2, 4]
        list2 = [1, 3, 4]
        l1 = ListNode.build_list_node(list1)
        l2 = ListNode.build_list_node(list2)

        list3 = [1, 1, 2, 3, 4, 4]
        expect = ListNode.build_list_node(list3)
        actual = _s.mergeTwoLists(l1, l2)

        self.assertEqual(expect, actual)

    def test_2(self):
        _s = leet.Solution()
        list1 = []
        list2 = []
        l1 = ListNode.build_list_node(list1)
        l2 = ListNode.build_list_node(list2)

        list3 = []
        expect = ListNode.build_list_node(list3)
        actual = _s.mergeTwoLists(l1, l2)

        self.assertEqual(expect, actual)

    def test_3(self):
        _s = leet.Solution()
        list1 = []
        list2 = [0]
        l1 = ListNode.build_list_node(list1)
        l2 = ListNode.build_list_node(list2)

        list3 = [0]
        expect = ListNode.build_list_node(list3)
        actual = _s.mergeTwoLists(l1, l2)

        self.assertEqual(expect, actual)


if __name__ == '__main__':
    unittest.main()
