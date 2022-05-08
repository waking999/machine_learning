from typing import Optional
from LeetCode.common.ListNode import ListNode


class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if list1 is None and list2 is None:
            return None

        rtn = ListNode()

        p1 = list1
        p2 = list2
        p3 = rtn
        while p1 is not None and p2 is not None:
            v1 = p1.val
            v2 = p2.val
            if v1 <= v2:
                p3.val = v1
                p1 = p1.next
            else:
                p3.val = v2
                p2 = p2.next

            p3.next = ListNode()
            p3 = p3.next

        while p1 is not None and p1.next is not None:
            v1 = p1.val
            p3.val = v1
            p3.next = ListNode()
            p3 = p3.next
            p1 = p1.next

        if p1 is not None and p1.next is None:
            p3.val = p1.val

        while p2 is not None and p2.next is not None:
            v1 = p2.val
            p3.val = v1
            p3.next = ListNode()
            p3 = p3.next
            p2 = p2.next

        if p2 is not None and p2.next is None:
            p3.val = p2.val

        return rtn
