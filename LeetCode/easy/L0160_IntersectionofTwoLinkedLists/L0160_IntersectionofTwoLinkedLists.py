from typing import Optional
from LeetCode.common.ListNode import ListNode


class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        if headA is None or headB is None:
            return None

        headALen = 1
        headBLen = 1

        headA1 = headA
        headB1 = headB

        while headA1.next is not None:
            headA1 = headA1.next
            headALen += 1

        while headB1.next is not None:
            headB1 = headB1.next
            headBLen += 1

        if headA1.val != headB1.val:
            return None

        diff = headALen - headBLen

        headA1 = headA
        headB1 = headB

        count = 0
        if diff > 0:
            while count < diff:
                headA1 = headA1.next;
                count += 1

        elif diff < 0:
            diff = 0 - diff
            while count < diff:
                headB1 = headB1.next
                count += 1

        while headA1 != headB1:
            headA1 = headA1.next
            headB1 = headB1.next

        return headA1

        return None
