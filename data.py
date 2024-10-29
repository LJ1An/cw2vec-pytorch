# -*- coding:UTF-8 -*-

import numpy
import random

# numpy.random.seed(12345)

'''
- 1、对每个单词进行序号化
    - 1、分词，zhihu.txt是输入数据，已经是分词好的
    - 2、获取每个token/单词的笔画信息，
    - 3、获取每个单词的笔画信息的n-gram排列
    - 4、对于n-gram排列，获取每个组合对于的索引id,此时每个单词都有[一组序号]去表示他了。而word2vec中每个单词是[一个序号]表示的
- 2、skip_gram模型，输入的是[bs, 363], 用一个363维度的向量表示单词。 然后进行embedding, 词汇表大小为3876(一共这么多种笔画的组合)
    emb_dimension为100，表示每个笔画向量映射成100的维度 变成[bs, 363, 100]。 然后进行求平均，变为[bs, 100]。
    同时有对应正样本单词为[bs,363] 进行embedding 变成[bs, 363,100] 求平均变为[bs, 100]。 
    然后上述两个相乘，然后sum(1), 然后求sigmod，然后求log。【感觉是算每个样本此时预测为正样本的概率对应的损失】
    
    然后算负样本的概率，他这里是先把特征去取相反数，然后去求的sigmod和log，得出来负样本的概率对应的损失
    
    最后返回正负样本的损失和
- 3、总体感觉代码中逻辑问题很多
    
    
    
    

'''

# 笔画编码映射 从111开始
stroke2id = {} # 长度为3875  id为-1表示没有笔画
for i in range(1, 6):
    for j in range(1, 6):
        for z in range(1, 6):
            stroke2id[str(i) + str(j) + str(z)] = len(stroke2id) + 1
            for m in range(1, 6):
                stroke2id[str(i) + str(j) + str(z) + str(m)] = len(stroke2id) + 1
                for n in range(1, 6):
                    stroke2id[str(i) + str(j) + str(z) + str(m) + str(n)] = len(stroke2id) + 1


class InputData:
    """
    存储数据，以及对数据进行笔画处理，生成批，进行负采样等操作
    """

    def __init__(self,
                 word2id,
                 id2word,
                 chchar2stroke,
                 file_name,
                 max_stroke,
                 n_sample,  # 5
                 word_frequency,
                 words_stroke_filename):
        self.input_file_name = file_name  # resource/data/zhihu.txt
        self.words_stroke_filename = words_stroke_filename  # resource/data/word_strokes.txt
        self.max_stroke = max_stroke
        self.word2id = word2id
        self.id2word = id2word
        self.n_sample = n_sample
        self.chchar2stroke = chchar2stroke
        self.word_count = len(self.word2id)
        self.word_frequency = word_frequency
        self.stroke2id = stroke2id
        self.get_wordid_to_strokeids(words_stroke_filename, max_stroke)
        self.init_sample_table()

    def get_word_strokeids(self,
                           word,
                           smallest_n=3,
                           biggest_n=5):
        '''
            word:输入的中文单词
            smallest_n:最小的n-gram
            biggest_n:最大的n-gram
            单词过来后，把单词拆开，遍历每个词，获得每个词的笔画信息，然后就可以获得这个word的笔画信息
            然后对笔画信息进行n-gram,eg: 大人：一ノ丶 ノ丶 表示为13434，
            3-gram：134、343、434
            4-gram：1343、3434
            5-gram：13434

            stroke为笔画的意思
        '''

        strokestr = ''  # 获得这个word的笔画信息，每个笔画是用数字表示的，eg: strokestr = 13434
        for ch in word:  # 遍历单词中的每个字
            if ch in self.chchar2stroke:  # resource/data/strokes.txt 如果字在笔画字典中，则获得字的笔画信息(由12345组成的字符串)
                strokestr += self.chchar2stroke[ch]  # 把笔画信息添加到strokestr中

        n_gram = []  # 存储所有的n-gram，此代码中是3 4 5的gram。 最终获得所有的strokestr的n-gram笔画对应的id值。(笔画对应是1 2 3 4 5，他们没有任何意义，只有id值才有意义，这样可以进行on-hot编码)
        # 此循环表示，获取所有的n-gram笔画信息对应的id值。
        # stroke2id 中存的是每中【笔画组合】对应的id值(这里的笔画组合长度是3到5)。  这里一共5中笔画，用1，2，3，4，5表示每一种笔画。 12345中任意三个组合(可重复 eg:111)表示3-gram的组合
        for i in range(smallest_n, biggest_n + 1):
            j = i
            while j <= len(strokestr):
                n_gram.append(stroke2id[strokestr[j - i:j]])
                j += 1

        return n_gram

    def get_wordid_to_strokeids(self,
                                words_stroke_filename,  # resource/data/word_strokes.txt
                                max_stroke):  # 363
        """
        创建词和笔画映射，笔画以max_stroke长度进行padding或截取，
        对于没有对应笔画的词，统一以-1对应的笔画来表示其笔画，其
        被初始化为[0,0,...,0]长度为max_stroke

        :param words_stroke_filename: 词和笔画对应文件
        :param max_stroke: 最大笔画n-gram数
        :return: 词索引和笔画索引映射
        """
        self.wordid2strokeids = {}
        # with open(words_stroke_filename, 'r', encoding='utf-8-sig') as fr:
        with open(words_stroke_filename, 'r', encoding='gbk') as fr:
            for i in fr.readlines():
                # 获取每一行，i中有三个元素 [单词，笔画, 笔画的n-gram组合(未映射到id的笔画组合)]
                i = i.strip().split("\t")
                # word2id为单词对应的id
                if i[0] in self.word2id: # TODO 注意这里是只对在单词表中的单词进行处理
                    # 字符串的list转换成实际的list
                    strokes = eval(i[2])
                    # 笔画组合映射成id
                    strokes_transform = [stroke2id[stroke] for stroke in strokes]
                    # 最大363种组合，如果长度小于363，则进行填充
                    if max_stroke > len(strokes_transform):
                        # 默认取3,4,5的n-gram词的最长特征为363，因此对长度不够的进行padding，填充0
                        strokes_transform = strokes_transform + [0] * (max_stroke - len(strokes_transform))
                    else:
                        # 多于363的截取
                        strokes_transform = strokes_transform[:max_stroke]
                    # 构造单词索引 和 对应的笔画n-gram组合。 注意word2id是从zhihu.txt构建的
                    self.wordid2strokeids[self.word2id[i[0]]] = strokes_transform

        # 当词在word2id中存在但在wordid2strokeids中不存在时统一用-1对应的笔画表示
        self.wordid2strokeids[-1] = [0] * max_stroke

        # 遍历词表，把单词id 和 词表中每个单词的n-gram的id值，映射起来。 TODO 感觉和上述重复了 ，猜测上述代码应该不需要写，应该是当初测试用的
        for word, id in self.word2id.items():
            # 如果词表中的单词，不在wordid2strokeids # TODO 单词肯定不在wordid2strokeids里啊  疑惑 应该是写 if id not in self.wordid2strokeids:
            if word not in self.wordid2strokeids:
                strokes_transform = self.get_word_strokeids(word)  # 获取单词的笔画的n-gram对应的id值
                if max_stroke > len(strokes_transform):
                    # 默认取3,4,5的n-gram词的最长特征为363，因此对长度不够的进行padding，填充0
                    strokes_transform = strokes_transform + [0] * (max_stroke - len(strokes_transform))
                else:
                    # 多于363的截取
                    strokes_transform = strokes_transform[:max_stroke]

                self.wordid2strokeids[id] = strokes_transform

    def init_sample_table(self):
        """
        初始化负抽样
        :return:
        """

        self.sample_table = []
        sample_table_size = 1e8
        # 词频变为词频的0.75次幂, 使得频率变低
        pow_frequency = numpy.array(list(self.word_frequency.values())) ** 0.75

        # 总的词频
        words_pow = sum(pow_frequency)

        # 每个词的概率
        self.ratio = pow_frequency / words_pow

        # 每个词的概率乘以10的8次方，进行四舍五入了，保留整数。  因为词的概率总和为1，所以乘以1e8,后，总和为1e8。表示如果有1e8总数，那么每个词的词频是多少
        count = numpy.round(self.ratio * sample_table_size)

        # sample_table为 [0,0,0,0,1,1,1,1,1,.....n,n,n,n] 每个词出现几次，则就有几个对应的id值
        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)  # 每个词的词频数目(基于1e8的词频数目)有多少个，则就有多少个这个词对应的id
        # sample_table的长度为99999963  因为精度和四舍五入问题，长度变为99999963，接近1e8
        self.sample_table = numpy.array(self.sample_table)

    def get_batch_pairs(self,
                        batch_size,  # 500
                        window_size,  # 5
                        word_data_ids,
                        shuffle=True):
        if shuffle:
            lst = list(range(len(word_data_ids)))
            random.shuffle(lst)
            word_data_ids = [word_data_ids[i] for i in lst]

        u_word_strokes = []
        v_word_strokes = []
        v_neg_strokes = []
        # 每行文本的单词数
        lens = [len(li) for li in word_data_ids]
        print("word numbers is:", sum(lens))
        # 遍历每个文本， 文本用单词表示，单词用id表示
        for k, linearr in enumerate(word_data_ids):
            # k为索引，linearr为文本的单词id列表
            len_line = len(linearr)
            # 遍历每一行文本 u为单词id
            for i, u in enumerate(linearr):
                # max(0, i - window_size) i为0到12 0-5 1-5 2-5 3-5 4-5 5-5 6-5...12-5    0+5 1+5 2+5 3+5 4+5 5+5 6+5...12+5 +1, 13
                # i=0 len_line = 8 则 range(0, 6)
                # i=1 len_line = 8 则 range(1, 7)
                # i=6 len_line = 8 则 range(1, 8)
                for j in range(max(0, i - window_size), min(i + window_size + 1, len_line)):
                    # u_word_strokes 单词的笔画n-gram组合,size为[500, 363] 表示批次500
                    #
                    if len(u_word_strokes) == batch_size:
                        yield u_word_strokes, v_word_strokes, v_neg_strokes
                        u_word_strokes = []
                        v_word_strokes = []
                        v_neg_strokes = []

                    if i == j:
                        continue
                    # 中心单词id对应的笔画 [500, 363]
                    u_word_strokes.append(self.wordid2strokeids[u])
                    # linearr[j]为中心词上下文单词中一个单词的笔画 [500, 363]
                    v_word_strokes.append(self.wordid2strokeids[linearr[j]])
                    # 五个负样本单词对应的笔画 [500, 5, 363]
                    v_neg = self.get_neg_v_neg_sampling(self.n_sample)
                    v_neg_strokes.append([self.wordid2strokeids[neg] for neg in v_neg])

        yield u_word_strokes, v_word_strokes, v_neg_strokes

    def get_neg_v_neg_sampling(self,
                               n_sample):
        """
        根据传入的参数进行负采样

        :param n_sample: 每个样本对应的采样数
        :return:
        """
        # 采样5个值，随机采列表中的值，因为列表中每个值出现的次数不一样，最终为采概率大的值。 最终返回5个id。id为单词的id值
        neg_v = numpy.random.choice(
            self.sample_table, size=n_sample).tolist()
        # neg_word_strokes = [[self.wordid2strokeids[j] for j in  i] for i in neg_v]
        # neg_word_p = [self.ratio[idx] for idx in neg_v]
        # neg_v shape:[batch_size, n_neg_sample], neg_word_p shape:[batch_size, n_neg_sample]
        return neg_v

    def evaluate_pair_count(self,
                            words_num,
                            window_size,
                            data_len):
        return words_num * (2 * window_size - 1) - (
                data_len - 1) * (1 + window_size) * window_size


if __name__ == '__main__':
    # data_path = "resource/data/word_strokes.txt.txt"
    # with open(data_path, 'r', encoding='GBK') as fr:
    #     for i in fr:
    #         print(i)

    # import chardet
    #
    # with open(data_path, 'rb') as f:
    #     result = chardet.detect(f.read())  # 读取一定量的数据进行编码检测
    #
    # print(result['encoding'])  # 打印检测到的编码
    # strokestr = '一ノ丶ノ丶'

    # strokestr = '13434'
    # n_gram = []
    # for i in range(3, 5 + 1):
    #     j = i
    #     # j为每个3、4、5的n-gram词，j为当前位置，j-i为当前位置的开始位置
    #     while j <= len(strokestr):
    #         # strokestr[0:3] strokestr[1:4] strokestr[2:5]
    #         # strokestr[0:4] strokestr[1:5]
    #         # strokestr[0:5]
    #         n_gram.append(stroke2id[strokestr[j - i:j]])
    #         print(strokestr[j - i:j])
    #         j += 1
    # print(n_gram)

    stroke2id = {} # 3875
    for i in range(1, 6):
        for j in range(1, 6):
            for z in range(1, 6):
                stroke2id[str(i) + str(j) + str(z)] = len(stroke2id) + 1
                for m in range(1, 6):
                    stroke2id[str(i) + str(j) + str(z) + str(m)] = len(stroke2id) + 1
                    for n in range(1, 6):
                        stroke2id[str(i) + str(j) + str(z) + str(m) + str(n)] = len(stroke2id) + 1
    print(len(stroke2id))
