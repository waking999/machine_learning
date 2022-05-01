from typing import Optional
from LeetCode.common.ListNode import ListNode


class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        f = head
        s = head
        while f is not None and s is not None and s.next is not None and f.next is not None and f.next.next is not None:
            s = s.next
            f = f.next.next

            if f == s:
                return True

        return False
