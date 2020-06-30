# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:02:30 2020

@author: 李稳
"""

import logging
import gensim
from gensim.models import word2vec
# 设置输出日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# 直接用gemsim提供的API去读取txt文件，读取文件的API有LineSentence 和 Text8Corpus, PathLineSentences等。
sentences = word2vec.LineSentence("choose_paper/data/ACM_dataset/abstracts.txt")
# 训练模型，词向量的长度设置为200， 迭代次数为8，采用skip-gram模型，模型保存为bin格式
model = gensim.models.Word2Vec(sentences, size=256, sg=1, iter=8)  
#textJaccard 
model.wv.save_word2vec_format('choose_paper/data/model/word2vec.txt',binary = False)
# 加载bin格式的模型
#wordVec = gensim.models.KeyedVectors.load_word2vec_format("model/word2Vec.bin", binary=True)