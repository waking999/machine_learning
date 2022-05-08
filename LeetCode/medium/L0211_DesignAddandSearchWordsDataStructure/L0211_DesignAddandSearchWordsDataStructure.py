class TrieNode:
    def __init__(self, c):
        self.char = c
        self.children = [None for i in range(26)]
        self.is_word = False


class WordDictionary:

    def __init__(self):
        self.root = TrieNode('0')
        self.aa = ord('a')

    def addWord(self, word: str) -> None:
        p = self.root
        for c in word:
            i=ord(c)- self.aa
            if p.children[i] is None:
                p.children[i] = TrieNode(c)
            p=p.children[i]
        p.is_word = True

    def search(self, word: str) -> bool:
        return self.match(word, 0, self.root)

    def match(self, word, index, node):
        if node is None:
            return False
        if index == len(word):
            return node.is_word
        if word[index] != '.':
            return node is not None and self.match(word, index + 1, node.children[ord(word[index])-self.aa])
        else:
            for child in node.children:
                if self.match(word, index + 1, child):
                    return True
        return False
