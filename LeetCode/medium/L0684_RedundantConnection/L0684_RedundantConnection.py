from typing import List


class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        if edges is None:
            return None

        parents = [i for i in range(len(edges) + 1)]

        def find_parent(i):
            if parents[i] != i:
                parents[i] = find_parent(parents[i])
            return parents[i]

        def set_parent(l, p):
            parents[l] = p

        for e in edges:
            u = find_parent(e[0])
            v = find_parent(e[1])
            if u == v:
                return e
            set_parent(u, v)
