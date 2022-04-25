class ListNode:
    def __init__(self, val=0, _next=None):
        self.val = val
        self.next = _next

    def __str__(self):
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
    def build_list_node(cls, _list):
        node = ListNode()
        p = node
        _l = len(_list)
        if _l == 0:
            return None

        for i in range(_l - 1):
            p.val = _list[i]
            p.next = ListNode()
            p = p.next

        p.val = _list[_l - 1]
        p.next = None
        return node
