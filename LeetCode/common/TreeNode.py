class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def _inorder(self):
        if self is None:
            return ''
        _str = ''
        if self.left is not None and self.left.val is not None:
            _str += self.left._inorder()
        _str += str(self.val)
        if self.right is not None and self.right.val is not None:
            _str += self.right._inorder()
        return _str

    def __str__(self):
        _str = self._inorder()
        return _str

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__str__() == other.__str__()
        else:
            return False

    @classmethod
    def _insert_node(cls, _list, _index, que):
        while len(que) > 0:
            tmp = que.pop(0)
            if tmp.left is None:
                if _index >= len(_list):
                    break
                v = _list[_index]
                if v is not None:
                    v_node = TreeNode(v)
                    tmp.left = v_node
                    que.append(v_node)
                else:
                    v_node = TreeNode(v)
                    tmp.left = v_node
                _index += 1
                if tmp.right is None:
                    if _index >= len(_list):
                        break
                    v = _list[_index]
                    if v is not None:
                        v_node = TreeNode(v)
                        tmp.right = v_node
                        que.append(v_node)
                    else:
                        v_node = TreeNode(v)
                        tmp.right = v_node
                    _index += 1
                else:
                    que.append(tmp.right)
            else:
                que.append(tmp.left)
                if tmp.right is None:
                    if _index >= len(_list):
                        break
                    v = _list[_index]
                    if v is not None:
                        v_node = TreeNode(v)
                        tmp.right = v_node
                        que.append(v_node)
                    else:
                        v_node = TreeNode(v)
                        tmp.right = v_node
                    _index += 1
                else:
                    que.append(tmp.right)

    @classmethod
    def build_tree_node(cls, _list):
        if _list is None:
            return None

        _l = len(_list)
        if _l == 0:
            return None

        root = TreeNode(_list[0])

        que = [root]
        cls._insert_node(_list, 1, que)

        return root

        # t_list = []
        # for val in _list:
        #     if val is not None:
        #         t_list.append(TreeNode(val))
        #     else:
        #         t_list.append(None)
        #
        # for i in range(_l):
        #     t_n = t_list[i]
        #     if t_n is not None:
        #         if i + 1 < _l:
        #             t_n.left = t_list[i + 1]
        #         if i + 2 < _l:
        #             t_n.right = t_list[i + 2]
        #
        # return t_list[0]
