# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:13:36 2020

@author: 李稳
"""


import numpy as np
import sys
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import itertools
from choose_paper.PaperCompare.rules import *



from tqdm import tqdm

 


dictAll0=np.load('choose_paper/PaperCompare/data_process_result/dictAll0.npy').item()
dictAll1=np.load('choose_paper/PaperCompare/data_process_result/dictAll1.npy').item()
dictAll2=np.load('choose_paper/PaperCompare/data_process_result/dictAll2.npy').item()
dictAll3=np.load('choose_paper/PaperCompare/data_process_result/dictAll3.npy').item()
dictAll4=np.load('choose_paper/PaperCompare/data_process_result/dictAll4.npy').item()


dict0 = np.load('choose_paper/PaperCompare/data_process_result/dict0.npy').item()
dict1 = np.load('choose_paper/PaperCompare/data_process_result/dict1.npy').item()
dict2 = np.load('choose_paper/PaperCompare/data_process_result/dict2.npy').item()
dict3 = np.load('choose_paper/PaperCompare//data_process_result/dict3.npy').item()  
dict4 = np.load('choose_paper/PaperCompare//data_process_result/dict4.npy').item()


def test2sequence(firstId,secondId,dicti):
    pad_array_first=dicti[firstId]
    pad_array_second=dicti[secondId]
    return pad_array_first,pad_array_second

def rulesEmbbeding(firstId,secondId,subId):
    firstRule=referenceJaccard(firstId,secondId)
    secondRule=PaperId2KeywordsJaccard(firstId,secondId)
    thirdRule=ccsSimilarity(firstId,secondId)
    fourthRule=textJaccard(firstId,secondId,subId)
    FourRules=[firstRule,secondRule,thirdRule,fourthRule]
    return FourRules

def predmodel(modelname,PaperIdList,dicti,dictAlli,subId):
    AllPaperPairs=list(itertools.combinations(PaperIdList, 2))
    first_list=[]
    second_list=[]
    FourRules_list=[]
    AllPaperPairs_len=len(AllPaperPairs)
    for i in tqdm(range(AllPaperPairs_len)):
        pad_array_first,pad_array_second=test2sequence(AllPaperPairs[i][0],AllPaperPairs[i][1],dicti)
        a=pad_array_first.tolist()[0]
        b=pad_array_second.tolist()[0]
        c=rulesEmbbeding(AllPaperPairs[i][0],AllPaperPairs[i][1],subId)
        first_list.append(a)
        second_list.append(b)
        FourRules_list.append(c)
        pass
    index_pad_array_first=np.array(first_list)
    index_pad_array_second=np.array(second_list)
    FourRules=np.array(FourRules_list)
    model=load_model(modelname)
    predlabel =model.predict([index_pad_array_first, index_pad_array_second, FourRules],
                                           batch_size=512, verbose=1)
    predlabel_list=predlabel.tolist()
    finalresult=[]
    for i in tqdm(range(len(AllPaperPairs))):
        if predlabel_list[i][0]>0.5:
            temp=AllPaperPairs[i]
            rules=rulesEmbbeding(temp[0],temp[1],subId)
            finalresult.append([temp,(dictAlli[temp[0]],dictAlli[temp[1]]),("reference",rules[0]),("keywords",rules[1]),("ccs",rules[2]),("Text Keywords",rules[3])])
        pass
    
    return finalresult

def predmodel_norules(modelname,PaperIdList,dicti,dictAlli,subId):
    AllPaperPairs=list(itertools.combinations(PaperIdList, 2))
    first_list=[]
    second_list=[]
    AllPaperPairs_len=len(AllPaperPairs)
    for i in tqdm(range(AllPaperPairs_len)):
        pad_array_first,pad_array_second=test2sequence(AllPaperPairs[i][0],AllPaperPairs[i][1],dicti)
        a=pad_array_first.tolist()[0]
        b=pad_array_second.tolist()[0]
        first_list.append(a)
        second_list.append(b)
        pass
    index_pad_array_first=np.array(first_list)
    index_pad_array_second=np.array(second_list)
    model=load_model(modelname)
    predlabel =model.predict([index_pad_array_first, index_pad_array_second],
                                           batch_size=512, verbose=1)
    predlabel_list=predlabel.tolist()
    finalresult=[]
    for i in tqdm(range(len(AllPaperPairs))):
        print(predlabel_list[i][0])
        if predlabel_list[i][0]>0.5:
            temp=AllPaperPairs[i]
            rules=rulesEmbbeding(temp[0],temp[1],subId)
            finalresult.append([temp,(dictAlli[temp[0]],dictAlli[temp[1]]),("reference",rules[0]),("keywords",rules[1]),("ccs",rules[2]),("Text Keywords",rules[3])])
        pass
    return finalresult

# PaperIdList=[261, 120856 ,206623, 282335 ,323946, 349912 ,360859, 365116 ,369284 ,384437 ,469528 ,472775 ,554076, 558733, 579836, 590915 ,618169, 623619, 739859, 819705, 928686, 1128659, 1326094 ,1333612, 1440142 ,1813120 ,2246947, 2418569 ,2719237 ,2811928, 2816699 ,2861761 ,2879837, 2902443, 2931902 ,2987096, 3015030 ,3032412, 3036835, 3152189 ,3292486 ,3366692 ,3545156 ,4037267 ,4037818, 4239368, 4254138]

# finalresult=predmodel("model/model4.h5",PaperIdList,dict4,dictAll4,4)
# #finalresult_=predmodel_norules("model_/model3.h5",PaperIdList,dict3,dictAll3,3)
# for i in range(len(finalresult)):
#     for j in range(len(finalresult[i])):
#         for m in range(len(finalresult[i][j])):
#             print(finalresult[i][j][m])


"""
print ('参数列表:', str(sys.argv))
argv_list=sys.argv
#第一个参数是：模型路径   第二个参数是：paperId的list   第三个参数是：子空间编号
modelname=argv_list[1]
PaperIdList=eval(argv_list[2])
subId=int(argv_list[3])
finalresult=predmodel(modelname,PaperIdList,subId)
print(finalresult)
"""