# -*- coding:UTF-8 -*-

import numpy
from collections import Counter

numpy.random.seed(12345)


def load_data(input_file_name, min_count, padding='<PAD>'):
    """
    加载数据，并生成数据集，词典，词频等
    :param input_file_name:  resource/data/zhihu.txt
    :param min_count: 5
    :param padding: '<PAD>'
    :return:
    """
    counter = Counter()
    words_num = 0  # 总单词数，不去重
    word_data_ids = []
    word_data_tmp = []
    with open(input_file_name, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            # 获得每一行文本中分好的单词[]
            line = line.strip().split(' ')
            # 获取单词数
            words_num += len(line)
            # 放入临时列表
            word_data_tmp.append(line)
            # 是对单词进行统计词频，eg:counter.update([a,b,a]) 统计出{'a':2,b:1}
            counter.update(line)

    word2id = dict()
    id2word = dict()
    wid = 0
    word_frequency = dict()
    # 遍历统计的词频，如果词频小于5，则不加入词典，同时减去该词的数量
    for w, c in counter.items():
        if c < min_count:  # 单词的词频小于min_count
            words_num -= c  # 在words_num中减去该词的数量, 相当于不统计这个单词了
            continue  # 跳过该单词，下边语句不执行，进行下一次循环
        word2id[w] = wid  # 构造单词和id的映射
        id2word[wid] = w  # 构造id和单词的映射
        word_frequency[wid] = c  # 构造单词id和词频的映射
        wid += 1  # id自增
    word2id[padding] = wid  # padding='<PAD>' 填充单词的id为最后一个
    id2word[wid] = padding  # 填充词的id和填充词的映射

    # 遍历每一行文本
    for linearr in word_data_tmp:
        # 通过单词映射的id，构造文本的id列表，如果单词不在词典中，则忽略这个单词。
        linearr_tmp = [word2id[word] for word in linearr if word in word2id]
        # 跳过空行，可能一行文本中，每个单词出现的次数都是小于min_count的5
        if len(linearr_tmp) > 0:
            # 把文本的id列表加入到word_data_ids中
            word_data_ids.append(linearr_tmp)

    # 返回  总单词数(不去重) 文本id列表， 单词id和词频的映射，单词和id映射，id和单词映射
    return words_num, word_data_ids, word_frequency, word2id, id2word


def load_strokes(stroke_path):
    """
    加载字对应的笔画，并转为论文中对应的编码
    :param stroke_path:
    :return:
    """
    stroke2id = {'横': '1', '提': '1', '竖': '2', '竖钩': '2', '撇': '3', '捺': '4', '点': '4'}
    chchar2stroke = {}

    with open(stroke_path, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            line = line.strip().split(':')
            if len(line) == 2:
                arr = line[1].split(',')
                strokes = [stroke2id[stroke] if stroke in stroke2id else '5' for stroke in arr]
                chchar2stroke[line[0]] = ''.join(strokes)

    return chchar2stroke


if __name__ == '__main__':
    # load_strokes('resource/data/strokes.txt')

    load_data('resource/data/zhihu.txt', 5)
