from typing import Optional
from LeetCode.common.ListNode import ListNode


class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None or head.next is None:
            return head

        p1 = head
        p2 = head.next
        while p1 is not None and p1.next is not None and p2 is not None:
            while p1 is not None and p2 is not None and p1.val == p2.val:
                p2 = p2.next
            p1.next = p2
            p1 = p2
            if p1 is not None:
                p2 = p1.next

        return head
