# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:57:17 2020

@author: 李稳
"""

import json  
import numpy as np
from choose_paper.data import config
from gensim import corpora, models, similarities

paperID_list=[]
with open(config.ARTICLES, 'r') as f:
    for index, line in enumerate(f.readlines()):
        line = line.strip('\n').split("\t")
        paperId=int(line[-1])
        paperID_list.append(paperId)
        


lines=[]
with open(config.SENTENCE_TEXTCNN, 'r') as f:
    for index, line in enumerate(f.readlines()):
        line = line.strip('\n')
        lines.append(line)
        


sentence_textcnn_list=[]
for eachline in lines:
    temp=json.loads(eachline)
    temp_arr=[]
    temp_arr.append(int(temp['paper_id']))
    temp_arr.append(int(temp['textcnn_label']))
    sentence_textcnn_list.append(temp_arr)
    
sentence_textcnn_array=np.array(sentence_textcnn_list)

PaperId_Subspace=[]
for each in paperID_list:
    temp_li=[each]
    temp={0:0,1:0,2:0,3:0,4:0}
    for id_label in sentence_textcnn_list:
        if  id_label[0]==each:
            if id_label[1]==0:
                temp[0]=temp[0]+1
            elif id_label[1]==1:
                temp[1]=temp[1]+1
            elif id_label[1]==2:
                temp[2]=temp[2]+1
            elif id_label[1]==3:
                temp[3]=temp[3]+1
            elif id_label[1]==4:
                temp[4]=temp[4]+1       
    temp_li.append(temp[0]) 
    temp_li.append(temp[1])     
    temp_li.append(temp[2]) 
    temp_li.append(temp[3])     
    temp_li.append(temp[4]) 
    PaperId_Subspace.append(temp_li)
 
    
PaperId_Subspace_dict={}
for each in PaperId_Subspace:
    temp_list=[]
    temp_list.append(each[1])
    temp_list.append(each[2])
    temp_list.append(each[3])
    temp_list.append(each[4])
    temp_list.append(each[5])    
    PaperId_Subspace_dict[each[0]]=temp_list
    
np.save('choose_paper/data/data/PaperId_Subspace_dict.npy', PaperId_Subspace_dict) 
PaperId_Subspace_dict = np.load('choose_paper/data/data/PaperId_Subspace_dict.npy').item()
  
    
    
    
    
    
    
    