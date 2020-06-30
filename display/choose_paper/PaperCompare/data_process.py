# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:44:28 2020

@author: 李稳
"""

############################################################################
##        一共有5个子空间
##         0：研究背景
##         1：研究问题
##         2：贡献
##         3：方法
##         4：实验
##
#############################################################################


import json  
import numpy as np
from choose_paper.data import config
from gensim import corpora, models, similarities

lines=[]
with open(config.SENTENCE_TEXTCNN, 'r') as f:
    for index, line in enumerate(f.readlines()):
        line = line.strip('\n')
        lines.append(line)
        

sentence_textcnn_list1=[]   
sentence_textcnn_list2=[] 
sentence_textcnn_list=[]
for eachline in lines:
    temp=json.loads(eachline)
    temp_arr1=[]
    temp_arr2=[]
    temp_arr=[]
    temp_arr1.append(int(temp['sentence_id']))
    temp_arr1.append(int(temp['paper_id']))
    temp_arr1.append(int(temp['textcnn_label']))
    temp_arr2.append(int(temp['sentence_id']))
    temp_arr2.append(temp['sentence_content'].strip('\n'))
    temp_arr.append(int(temp['sentence_id']))
    temp_arr.append(int(temp['paper_id']))
    temp_arr.append(int(temp['textcnn_label']))
    temp_arr.append(temp['sentence_content'].strip('\n'))
    sentence_textcnn_list1.append(temp_arr1)
    sentence_textcnn_list2.append(temp_arr2)
    sentence_textcnn_list.append(temp_arr)
    
sentence_textcnn_array1=np.array(sentence_textcnn_list1)
sentence_textcnn_array2=np.array(sentence_textcnn_list2)


def Paperid2SubspaceAllSentences(sentence_textcnn_list,subId):
    SubSpace_dict={}
    paper_id_list=[]
    for each in sentence_textcnn_list:
        if each[-2]==subId:
            paper_id=each[1]
            if paper_id in paper_id_list:
                SubSpace_dict[paper_id]=SubSpace_dict[paper_id]+each[-1]
            else:
                SubSpace_dict[paper_id]=each[-1]
            paper_id_list.append(paper_id)
                
    return SubSpace_dict    
            




class DocumentSimilar(object):
    def __init__(self, documents):
        self.documents = documents
        self.dictionary = None
        self.tfidf = None
        self.similar_matrix = None
        self.calculate_similar_matrix()

    @staticmethod
    def split_word(document):
        """
        分词，去除停用词
        """
        text=document.split(" ")

        return text

    def calculate_similar_matrix(self):
        """
        计算相似度矩阵及一些必要数据
        """
        words = [self.split_word(document) for document in self.documents]

        self.dictionary = corpora.Dictionary(words)
        corpus = [self.dictionary.doc2bow(word) for word in words]
        self.tfidf = models.TfidfModel(corpus)
        corpus_tfidf = self.tfidf[corpus]
     #   self.similar_matrix = similarities.MatrixSimilarity(corpus_tfidf)
        self.similar_matrix = similarities.Similarity("",corpus_tfidf,len(self.dictionary))

    def get_similar(self, document):
        """
        计算要比较的文档与语料库中每篇文档的相似度
        """
        words = self.split_word(document)
        corpus = self.dictionary.doc2bow(words)
        corpus_tfidf = self.tfidf[corpus]
        return self.similar_matrix[corpus_tfidf]
    
  
    
def SecMax(list):
    list.sort()
    count=list.count(list[len(list) - 1] )
    c = 0
    while c < count:
        list.pop()
        c+=1
    return list[len(list) - 1]    
    
    


def train_pairs_pos_neg(SubSpace_dict):
    documents=[]
    index_list=[]
    for key,value in SubSpace_dict.items():
        index_list.append(key)
        documents.append(value)
    SubSpace_train_pairs=[]
    SubSpace_train_pairs_=[]
    doc_similar=DocumentSimilar(documents)
    for key,value in SubSpace_dict.items():
        a=key
        temp=list(doc_similar.get_similar(value))
        temp_=temp.copy()
        Second_Num=SecMax(temp_)
        min_num=min(temp)
        Second_Maxnum_Index=temp.index(Second_Num)
        min_num_index=temp.index(min_num)
        c=index_list[min_num_index]
        b=index_list[Second_Maxnum_Index]
        SubSpace_train_pairs.append([a,b,Second_Num])
        SubSpace_train_pairs_.append([a,c,min_num])
    SubSpace_train_pairs=np.array(SubSpace_train_pairs)
    SubSpace_train_pairs_=np.array(SubSpace_train_pairs_)
    return SubSpace_train_pairs,SubSpace_train_pairs_








def Max100_pos(SubSpace_train_pairs_pos):
    list_=[each[-1] for each in SubSpace_train_pairs_pos]
    array=np.array(list_)
    Max100=list(array.argsort()[-100:][::-1])
    Max100_pos=[]
    for each in Max100:
        temp=[]
        temp_li=SubSpace_train_pairs_pos[int(each)]
        temp.append(int(temp_li[0]))
        temp.append(int(temp_li[1]))
        Max100_pos.append(temp)
    return Max100_pos


def Max100_neg(SubSpace_train_pairs_neg):
    list_=[each[-1] for each in SubSpace_train_pairs_neg]
    array_=np.array(list_)
    Min100_=list(array_.argsort()[:100][::])
    Min100_neg=[]
    for each in Min100_:
        temp=[]
        temp_li=SubSpace_train_pairs_neg[int(each)]
        temp.append(int(temp_li[0]))
        temp.append(int(temp_li[1]))
        Min100_neg.append(temp)
    return Min100_neg

"""
SubSpace0_dict=Paperid2SubspaceAllSentences(sentence_textcnn_list,0)
SubSpace1_dict=Paperid2SubspaceAllSentences(sentence_textcnn_list,1)
SubSpace2_dict=Paperid2SubspaceAllSentences(sentence_textcnn_list,2)
SubSpace3_dict=Paperid2SubspaceAllSentences(sentence_textcnn_list,3)
SubSpace4_dict=Paperid2SubspaceAllSentences(sentence_textcnn_list,4)

np.save('data_process_result/SubSpace0_dict.npy', SubSpace0_dict) 
np.save('data_process_result/SubSpace1_dict.npy', SubSpace1_dict) 
np.save('data_process_result/SubSpace2_dict.npy', SubSpace2_dict) 
np.save('data_process_result/SubSpace3_dict.npy', SubSpace3_dict) 
np.save('data_process_result/SubSpace4_dict.npy', SubSpace4_dict)  

SubSpace0_dict = np.load('data_process_result/SubSpace0_dict.npy').item()
SubSpace1_dict = np.load('data_process_result/SubSpace1_dict.npy').item()
SubSpace2_dict = np.load('data_process_result/SubSpace2_dict.npy').item()
SubSpace3_dict = np.load('data_process_result/SubSpace3_dict.npy').item()  
SubSpace4_dict = np.load('data_process_result/SubSpace4_dict.npy').item()




SubSpace0_train_pairs,SubSpace0_train_pairs_=train_pairs_pos_neg(SubSpace0_dict)
SubSpace1_train_pairs,SubSpace1_train_pairs_=train_pairs_pos_neg(SubSpace1_dict)
SubSpace2_train_pairs,SubSpace2_train_pairs_=train_pairs_pos_neg(SubSpace2_dict)
SubSpace3_train_pairs,SubSpace3_train_pairs_=train_pairs_pos_neg(SubSpace3_dict)
SubSpace4_train_pairs,SubSpace4_train_pairs_=train_pairs_pos_neg(SubSpace4_dict)

np.save('data_process_result/SubSpace0_train_pairs+.npy',SubSpace0_train_pairs)
np.save('data_process_result/SubSpace0_train_pairs-.npy',SubSpace0_train_pairs_)
np.save('data_process_result/SubSpace1_train_pairs+.npy',SubSpace1_train_pairs)
np.save('data_process_result/SubSpace1_train_pairs-.npy',SubSpace1_train_pairs_)
np.save('data_process_result/SubSpace2_train_pairs+.npy',SubSpace2_train_pairs)
np.save('data_process_result/SubSpace2_train_pairs-.npy',SubSpace2_train_pairs_)
np.save('data_process_result/SubSpace3_train_pairs+.npy',SubSpace3_train_pairs)
np.save('data_process_result/SubSpace3_train_pairs-.npy',SubSpace3_train_pairs_)
np.save('data_process_result/SubSpace4_train_pairs+.npy',SubSpace4_train_pairs)
np.save('data_process_result/SubSpace4_train_pairs-.npy',SubSpace4_train_pairs_)


SubSpace0_train_pairs=np.load('data_process_result/SubSpace0_train_pairs+.npy')
SubSpace1_train_pairs=np.load('data_process_result/SubSpace1_train_pairs+.npy')
SubSpace2_train_pairs=np.load('data_process_result/SubSpace2_train_pairs+.npy')
SubSpace3_train_pairs=np.load('data_process_result/SubSpace3_train_pairs+.npy')
SubSpace4_train_pairs=np.load('data_process_result/SubSpace4_train_pairs+.npy')


SubSpace0_train_pairs=SubSpace0_train_pairs.tolist()
SubSpace1_train_pairs=SubSpace1_train_pairs.tolist()
SubSpace2_train_pairs=SubSpace2_train_pairs.tolist()
SubSpace3_train_pairs=SubSpace3_train_pairs.tolist()
SubSpace4_train_pairs=SubSpace4_train_pairs.tolist()

SubSpace0_train_pairs_=np.load('data_process_result/SubSpace0_train_pairs-.npy')
SubSpace1_train_pairs_=np.load('data_process_result/SubSpace1_train_pairs-.npy')
SubSpace2_train_pairs_=np.load('data_process_result/SubSpace2_train_pairs-.npy')
SubSpace3_train_pairs_=np.load('data_process_result/SubSpace3_train_pairs-.npy')
SubSpace4_train_pairs_=np.load('data_process_result/SubSpace4_train_pairs-.npy')

SubSpace0_train_pairs_=SubSpace0_train_pairs_.tolist()
SubSpace1_train_pairs_=SubSpace1_train_pairs_.tolist()
SubSpace2_train_pairs_=SubSpace2_train_pairs_.tolist()
SubSpace3_train_pairs_=SubSpace3_train_pairs_.tolist()
SubSpace4_train_pairs_=SubSpace4_train_pairs_.tolist()



Max100_0_pos=Max100_pos(SubSpace0_train_pairs)
Min100_0_neg=Max100_neg(SubSpace0_train_pairs_)
Max100_1_pos=Max100_pos(SubSpace1_train_pairs)
Min100_1_neg=Max100_neg(SubSpace1_train_pairs_)
Max100_2_pos=Max100_pos(SubSpace2_train_pairs)
Min100_2_neg=Max100_neg(SubSpace2_train_pairs_)
Max100_3_pos=Max100_pos(SubSpace3_train_pairs)
Min100_3_neg=Max100_neg(SubSpace3_train_pairs_)
Max100_4_pos=Max100_pos(SubSpace4_train_pairs)
Min100_4_neg=Max100_neg(SubSpace4_train_pairs_)
   
Max100_0_pos_array=np.array(Max100_0_pos)  
Min100_0_neg_array=np.array(Min100_0_neg)  
np.save('data_process_result/Max100_0_pos_array.npy',Max100_0_pos_array)
np.save('data_process_result/Min100_0_neg_array.npy',Min100_0_neg_array)

Max100_1_pos_array=np.array(Max100_1_pos)  
Min100_1_neg_array=np.array(Min100_1_neg)  
np.save('data_process_result/Max100_1_pos_array.npy',Max100_1_pos_array)
np.save('data_process_result/Min100_1_neg_array.npy',Min100_1_neg_array)

Max100_2_pos_array=np.array(Max100_2_pos)  
Min100_2_neg_array=np.array(Min100_2_neg)  
np.save('data_process_result/Max100_2_pos_array.npy',Max100_2_pos_array)
np.save('data_process_result/Min100_2_neg_array.npy',Min100_2_neg_array)

Max100_3_pos_array=np.array(Max100_3_pos)  
Min100_3_neg_array=np.array(Min100_3_neg)  
np.save('data_process_result/Max100_3_pos_array.npy',Max100_3_pos_array)
np.save('data_process_result/Min100_3_neg_array.npy',Min100_3_neg_array)

Max100_4_pos_array=np.array(Max100_4_pos)  
Min100_4_neg_array=np.array(Min100_4_neg)  
np.save('data_process_result/Max100_4_pos_array.npy',Max100_4_pos_array)
np.save('data_process_result/Min100_4_neg_array.npy',Min100_4_neg_array)

"""