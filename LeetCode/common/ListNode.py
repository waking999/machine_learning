class ListNode:
    def __init__(self, val=0, _next=None):
        self.val = val
        self.next = _next

    def _has_cycle(self):
        f = self
        s = self
        while f is not None and s is not None and s.next is not None and f.next is not None and f.next.next is not None:
            s = s.next
            f = f.next.next

            if f == s:
                return True

        return False

    def __str__(self):
        if self._has_cycle():
            return ''

        p = self
        _str = ''
        while p is not None and p.next is not None:
            _str += str(p.val) + '->'
            p = p.next

        if p is not None:
            _str += str(p.val)

        return _str

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__str__() == other.__str__()
        else:
            return False

    @classmethod
    def build_list_node(cls, _list, pos=-1):
        _l = len(_list)
        if _l == 0:
            return None

        node = ListNode()
        p = node

        p_cycle_node = None
        for i in range(_l - 1):
            p.val = _list[i]
            if i == pos:
                p_cycle_node = p
            p.next = ListNode()
            p = p.next

        p.val = _list[_l - 1]
        p.next = p_cycle_node
        return node
