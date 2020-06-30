# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 11:55:18 2020

@author: 李稳
"""

import numpy as np
import sys
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import itertools
from choose_paper.data.rules import *
from choose_paper.data import config
SubSpace0_dict = np.load('choose_paper/data/data_process_result/SubSpace0_dict.npy').item()
SubSpace1_dict = np.load('choose_paper/data/data_process_result/SubSpace1_dict.npy').item()
SubSpace2_dict = np.load('choose_paper/data/data_process_result/SubSpace2_dict.npy').item()
SubSpace3_dict = np.load('choose_paper/data/data_process_result/SubSpace3_dict.npy').item()  
SubSpace4_dict = np.load('choose_paper/data/data_process_result/SubSpace4_dict.npy').item()
paperID_list=[]
with open(config.ARTICLES, 'r') as f:
    for index, line in enumerate(f.readlines()):
        line = line.strip('\n').split("\t")
        paperId=int(line[-1])
        paperID_list.append(paperId)   
        
"""
def paperId2sequence(paperId,SubSpace_dict):
    word_index = np.load('data_process_result/word_index.npy').item()
    index_list=list(word_index.keys())
    subspace_keys=SubSpace_dict.keys()
    if paperId in subspace_keys:
        temp_str1=SubSpace_dict[paperId]
        temp_list_word1=temp_str1.split(" ")
        temp_list_index=[]    
        for i in temp_list_word1:
            if i in index_list:
                temp_list_index.append(word_index[i])
        pad_array = pad_sequences([temp_list_index], maxlen=150) 
    elif paperId not in subspace_keys:
        pad_array=np.array([[0]*150])
    return pad_array
dict0={}
dict1={}
dict2={}
dict3={}
dict4={}
for each in paperID_list:
    dict0[each]=paperId2sequence(each,SubSpace0_dict)
    dict1[each]=paperId2sequence(each,SubSpace1_dict)
    dict2[each]=paperId2sequence(each,SubSpace2_dict)
    dict3[each]=paperId2sequence(each,SubSpace3_dict)
    dict4[each]=paperId2sequence(each,SubSpace4_dict)
np.save('data_process_result/dict0.npy', dict0) 
np.save('data_process_result/dict1.npy', dict1) 
np.save('data_process_result/dict2.npy', dict2) 
np.save('data_process_result/dict3.npy', dict3) 
np.save('data_process_result/dict4.npy', dict4) 
"""

a=SubSpace0_dict.keys()
b=SubSpace1_dict.keys()
c=SubSpace2_dict.keys()
d=SubSpace3_dict.keys()
e=SubSpace4_dict.keys()
dictAll0={}
dictAll1={}
dictAll2={}
dictAll3={}
dictAll4={}


for each in paperID_list:
    if each in a:
        dictAll0[each]=SubSpace0_dict[each]
    elif each not in a:
        dictAll0[each]=None       
    if each in b:
        dictAll1[each]=SubSpace1_dict[each]
    elif each not in b:
        dictAll1[each]=None
    if each in c:
        dictAll2[each]=SubSpace2_dict[each]
    elif each not in c:
        dictAll2[each]=None
    if each in d:
        dictAll3[each]=SubSpace3_dict[each]
    elif each not in d:
        dictAll3[each]=None
    if each in e:
        dictAll4[each]=SubSpace4_dict[each]
    elif each not in e:
        dictAll4[each]=None        
np.save('choose_paper/data/data_process_result/dictAll0.npy', dictAll0) 
np.save('choose_paper/data/data_process_result/dictAll1.npy', dictAll1) 
np.save('choose_paper/data/data_process_result/dictAll2.npy', dictAll2) 
np.save('choose_paper/data/data_process_result/dictAll3.npy', dictAll3) 
np.save('choose_paper/data/data_process_result/dictAll4.npy', dictAll4) 
    