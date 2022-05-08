import unittest
import L0208_ImplementTrie as leet


class MyTestCase(unittest.TestCase):
    def test_1(self):
        trie = leet.Trie()
        trie.insert("apple")
        e1 = True
        a1 = trie.search("apple")
        self.assertEqual(e1, a1)
        e2 = False
        a2 = trie.search("app")
        self.assertEqual(e2, a2)
        e3 = True
        a3 = trie.startsWith("app")
        self.assertEqual(e3, a3)
        trie.insert("app")
        e4 = True
        a4 = trie.search("app")
        self.assertEqual(e4, a4)


if __name__ == '__main__':
    unittest.main()
