#!/usr/bin/env python
#coding=utf-8

# Copyright(c) 2013 python-sparselda project.
# Author: Lifeng Wang (ofandywang@gmail.com)

class Vocabulary(object):
    """Vocabulary implements the map-structure of words.
    """
    def __init__(self):
        self.word_map = {}  # item fmt: word -> index
        self.words = []  # item fmt: word, default index

    def load(self, filename):
        """read words from filename.
        line fmt: word [\t count]
        """
        self.word_map.clear()
        self.words = []
        fp = open(filename, 'r')
        for line in fp.readlines():
            line = line.decode('gbk')
            fields = line.strip().split('\t')
            if len(fields) > 0 and fields[0] not in self.word_map:
                self.word_map[fields[0]] = len(self.words)
                self.words.append(fields[0])
        fp.close()

    def has_word(self, word):
        return (word in self.word_map)

    def word_index(self, word):
        return self.word_map.get(word, -1)

    def word(self, index):
        assert index >= 0 and index < len(self.words)
        return self.words[index]

    def size(self):
        return len(self.words)

