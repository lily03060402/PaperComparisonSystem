# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:27:32 2020

@author: 李稳
"""

from choose_paper.PaperCompare import config

import json
from numpy import *
import numpy as np
from gensim.models import word2vec
import gensim
import numpy as np
from scipy.linalg import norm
from numpy import *

#keywords

# 获取数据
word_dic = np.load('choose_paper/PaperCompare//data_process_result/word_dic.npy', allow_pickle=True).item()
# 获取词向量
model1 = gensim.models.KeyedVectors.load_word2vec_format("choose_paper/PaperCompare/data_process_result/keyword.txt", binary=False)

# 定义相似度函数
def PaperId2KeywordsJaccard(id1, id2):
    simi = 0
    id1 = str(int(id1))
    id2 = str(int(id2))
    keywords1 = word_dic[id1]
    keywords2 = word_dic[id2]
    len1 = len(keywords1)
    len2 = len(keywords2)
    max_dis = []
    if len1 > len2 > 0:
        for key1 in keywords1:
            # 每一个关键词与另一组关键词求cosine distance
            dis_list = []
            for key2 in keywords2:
                v1 = model1[key1]
                v2 = model1[key2]
                s = np.dot(v1, v2) / (norm(v1) * norm(v2))
                dis_list.append(s)
            maxc = max(dis_list)
            max_dis.append(maxc)
        if len(max_dis)==0:
            return 0
        simi = mean(max_dis)
    elif len1 > 0:
        for key2 in keywords2:
            # 每一个关键词与另一组关键词求cosine distance
            dis_list = []
            for key1 in keywords1:
                v1 = model1[key1]
                v2 = model1[key2]
                s = np.dot(v1, v2) / (norm(v1) * norm(v2))
                dis_list.append(s)
            maxc = max(dis_list)
            max_dis.append(maxc)
        if len(max_dis)==0:
            return 0
        simi = mean(max_dis)
    return simi
"""
def PaperId2KeywordsJaccard(paperId1,paperId2):
    paperID_list=[]
    with open(config.ARTICLES, 'r') as f:
        for index, line in enumerate(f.readlines()):
            line = line.strip('\n').split("\t")
            paperId=int(line[-1])
            paperID_list.append(paperId)
    paperId_keywords_dict={}
    index=0
    with open(config.KEYWORDS,'r') as f:
        for line in f:
            line=line.strip('\n').split("\t")
            line=line[:-1]
            paperId=paperID_list[index]
            index=index+1
            paperId_keywords_dict[paperId]=line 
    temp_list1=paperId_keywords_dict[int(paperId1)]
    temp_list2=paperId_keywords_dict[int(paperId2)]
    temp_set=set(temp_list1+temp_list2)
    fenmu=len(temp_set)+0.000001
    fenzi=len(temp_list1)+len(temp_list2)-fenmu
    return fenzi/fenmu
    
"""
#refference
with open("choose_paper/PaperCompare/data_process_result/link.json", 'r') as json_file:
    link=json.load(json_file)
    
def referenceJaccard(id1,id2):
    if id1==id2:
        return 1
    temp_list=link.keys()
    if id1 not in temp_list or id2 not in temp_list:
        return 0
    links1=set(link[id1])
    links2=set(link[id2])
    if links1.intersection(links2)==None or links1.intersection_update(links2)==None:
        return 0
    else:
        jaccard=len(links1.intersection(links2))/(len(links1.intersection_update(links2))+1)#借鉴推荐系统上惩罚热门的做法？？？
        return jaccard
#ccs
        
    
def fun1(tx,ty):##tx,ty的形式为'f.2.2'，只包含属性代表
    if(tx==ty):
        return 1
    XT=tx[0]
    YT=ty[0]
    #print(XT,YT)
    if(XT!=YT):
        return 0##直接返回相似度，不是距离，距离算无穷大
    else :
        lx=len(tx)
        ly=len(ty)
        if(lx>ly):##保证lx<ly
            tt=lx
            lx=ly
            ly=tt
            tt=tx
            tx=ty
            ty=tt##数值和字符串都换回来
        if(lx==ly):
            if(lx==1):
                return 1/1
            elif(lx==3):
                return 1/4
            else:
                #print(lx)
#                 xx=tx
#                 yy=ty##(???，迷惑性为)
                ##xx=='f.2.2'
                mx=tx.split(".")[1]
                my=ty.split(".")[1]##找出第二位
                if(mx==my):
                    return 1/4
                else:
                     return 1/8
        else:##lx<ly
            if(lx==1)&(ly==3):
                return 1/2
            elif(lx==1)&(ly==5):
                return 1/4
            else:
#                 xx=tx
#                 yy=ty
                mx=tx.split(".")[1]
                my=ty.split(".")[1]##找出第二位
                if(mx==my):
                    return 1/4
                else:
                     return 1/6
                 
"""                    
listid = []
for line in open("data_process_result/treenode.txt","r"): #设置文件对象并读取每一行文件
    listid.append(line.replace("\n","")) 
def ccsSimilarity(X,Y):
    X=int(X)
    Y=int(Y)##ID i
    if X>43380 or Y>43380:
        return 0
    tx=listid[X].split(",")
    ty=listid[Y].split(",")
    ##考虑多于一个分类的情况
    lx=len(tx)
    ly=len(ty)
    if(lx>2)|(ly>2):
        if(lx>2)&(ly <=2):
            r=[]
            for i in range(1,lx):
                r.append(fun1(tx[i],ty[1]))
            return max(r)###取最大值
        elif(lx<=2)&(ly>2):
            r=[]
            for i in range(1,ly):
                r.append(fun1(tx[1],ty[i]))
            return max(r)###取最大值
        else:##两个都是多分类
            r=[]
            for i in range(1,lx):
                for j in range(1,ly):
                    r.append(fun1(tx[i],ty[j]))
            return max(r)###取最大值
                
    else:
        res=fun1(tx[1],ty[1])
    return res


"""
def ccsSimilarity(X,Y):
    listid = []
    for line in open("choose_paper/PaperCompare/data_process_result/treenode.txt","r"): #设置文件对象并读取每一行文件
        listid.append(line.replace("\n","")) 
    X=int(X)
    Y=int(Y)##ID i
    if X==Y:
        return 1
    if X>43380 or Y>43380:
        return 0
    tx=listid[X].split(",")
    ty=listid[Y].split(",")
    if(len(tx[1])==0) or (len(ty[1])==0):
        return 0
    XT=tx[1][0]
    YT=ty[1][0]
    #print(XT,YT)
    if(XT!=YT):
        return 0##直接返回相似度，不是距离，距离算无穷大
    else :
        lx=len(tx[1])
        ly=len(ty[1])
        if(lx>ly):##保证lx<ly
            tt=lx
            lx=ly
            ly=tt
            tt=tx
            tx=ty
            ty=tt##数值和字符串都换回来
        if(lx==ly):
            if(lx==1):
                return 1/1
            elif(lx==3):
                return 1/4
            else:
                #print(lx)
                xx=tx[1]
                yy=ty[1]
                mx=xx.split(".")[1]
                my=yy.split(".")[1]##找出第二位
                if(mx==my):
                    return 1/4
                else:
                     return 1/8
        else:##lx<ly
            if(lx==1)&(ly==3):
                return 1/2
            elif(lx==1)&(ly==5):
                return 1/4
            else:
                xx=tx[1]
                yy=ty[1]
                mx=xx.split(".")[1]
                my=yy.split(".")[1]##找出第二位
                if(mx==my):
                    return 1/4
                else:
                     return 1/6

#textJecard 

                 
model = gensim.models.KeyedVectors.load_word2vec_format("choose_paper/PaperCompare/data_process_result/subkey.txt", binary=False)
vocab_li=(model.wv.vocab).keys()
key_dict0 = np.load("choose_paper/PaperCompare/data_process_result/sub0_keyword.npy", allow_pickle=True).item()
key_dict1 = np.load("choose_paper/PaperCompare/data_process_result/sub1_keyword.npy", allow_pickle=True).item()
key_dict2 = np.load("choose_paper/PaperCompare/data_process_result/sub2_keyword.npy", allow_pickle=True).item()
key_dict3 = np.load("choose_paper/PaperCompare/data_process_result/sub3_keyword.npy", allow_pickle=True).item()
key_dict4 = np.load("choose_paper/PaperCompare/data_process_result/sub4_keyword.npy", allow_pickle=True).item()
def textJaccard(id1,id2,subid):
    if subid==0:
        key_dict=key_dict0
    elif subid==1:
        key_dict=key_dict1
    elif subid==2:
        key_dict=key_dict2
    elif subid==3:
        key_dict=key_dict3
    elif subid==4:
        key_dict=key_dict4
    simi = 0
    if id1 in key_dict.keys()and id2 in key_dict.keys():
        keywords1 = key_dict[id1]-{'-'}
        keywords2 = key_dict[id2]-{'-'}
        len1 = len(keywords1)
        len2 = len(keywords2)
        max_dis = []
        if len1>len2>0:
            for key1 in keywords1:
                # 每一个关键词与另一组关键词求cosine distance
                dis_list = []
                for key2 in keywords2:
                    if key1 in vocab_li and key2 in vocab_li:
                        v1 = model[key1]
                        v2 = model[key2]
                        s = np.dot(v1, v2) / (norm(v1) * norm(v2))
                        dis_list.append(s)
                if dis_list!=[]:
                    maxc = max(dis_list)
                    max_dis.append(maxc)
            if len(max_dis)==0:
                return 0
            simi = mean(max_dis)
        elif len1>0:
            for key2 in keywords2:
                # 每一个关键词与另一组关键词求cosine distance
                dis_list = []
                for key1 in keywords1:
                    if key1 in vocab_li and key2 in vocab_li:
                        v1 = model[key1]
                        v2 = model[key2]
                        s = np.dot(v1, v2) / (norm(v1) * norm(v2))
                        dis_list.append(s)
                if dis_list!=[]:
                    maxc = max(dis_list)
                    max_dis.append(maxc)
            if len(max_dis)==0:
                return 0
            simi = mean(max_dis)
    else:
        return 0
    return simi
"""
import numpy as np
def textJaccard(id1,id2,subid):
    filename = 'data_process_result/''sub'+str(subid)+'_keyword.npy'
    key_dict = np.load(filename, allow_pickle=True).item()
    if id1 in key_dict.keys()and id2 in key_dict.keys():
        keyword1 = key_dict[id1]
        keyword2 = key_dict[id2]
        temp = 0
        for i in keyword1:
            if i in keyword2:
                temp = temp+1
        fenmu = len(keyword1)+len(keyword2)-temp+0.00001#并集
        jaccard_coefficient = float(temp/fenmu)
    else:
        jaccard_coefficient = 0
    return jaccard_coefficient

"""
    
