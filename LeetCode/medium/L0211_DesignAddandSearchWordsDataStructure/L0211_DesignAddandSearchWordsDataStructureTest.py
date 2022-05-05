import unittest
import L0211_DesignAddandSearchWordsDataStructure as leet


class MyTestCase(unittest.TestCase):
    def test_something(self):
        wordDictionary = leet.WordDictionary()
        wordDictionary.addWord("bad")
        wordDictionary.addWord("dad")
        wordDictionary.addWord("mad")
        e1 = False
        a1 = wordDictionary.search("pad")
        self.assertEqual(e1, a1)
        e2 = True
        a2 = wordDictionary.search("bad")
        self.assertEqual(e2, a2)
        e3 = True
        a3 = wordDictionary.search(".ad")
        self.assertEqual(e3, a3)

        e4 = True
        a4 = wordDictionary.search("b..")
        self.assertEqual(e4, a4)


if __name__ == '__main__':
    unittest.main()
