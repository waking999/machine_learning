class TrieNode:
    def __init__(self, c):
        self.char = c
        self.children = [None for i in range(26)]
        self.is_word = False


class Trie:


    def __init__(self):
        self.root = TrieNode('0')
        self.aa = ord("a")

    def insert(self, word: str) -> None:
        p = self.root

        for c in word:
            if p.children[ord(c) - self.aa] is None:
                p.children[ord(c) - self.aa] = TrieNode(c)
            p = p.children[ord(c) - self.aa]
        p.is_word = True

    def search(self, word: str) -> bool:

        p = self.root
        for c in word:
            if p.children[ord(c) - self.aa] is None:
                return False
            p = p.children[ord(c) - self.aa]
        return p.is_word

    def startsWith(self, prefix: str) -> bool:

        p = self.root
        for c in prefix:
            if p.children[ord(c) - self.aa] is None:
                return False
            p = p.children[ord(c) - self.aa]
        return True
