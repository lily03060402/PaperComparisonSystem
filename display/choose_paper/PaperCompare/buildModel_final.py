# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 08:52:11 2020

@author: 李稳
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:49:18 2020

@author: 李稳
"""
import config
import time
import pickle
import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.layers import Embedding, merge, Reshape, Activation, RepeatVector, Permute, Lambda, GlobalMaxPool1D, \
    concatenate
from keras import initializers
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.layers import Dense, Conv1D, MaxPooling1D, Input, Flatten, Dropout, Concatenate, LSTM, Bidirectional, GRU
from keras.metrics import categorical_accuracy,mean_squared_error,binary_accuracy
from keras.models import Model
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
import random
from choose_paper.data.rules import *






def embeddingMatrix():
    print('word embedding')
    embeddings_index = {}
    word_index={}
    embedding_max_value = 0
    embedding_min_value = 1
    i=1
    with open(config.WORD_EMBEDDING_DIR, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            if len(line) != 257:
                word_num=int(line[0])
                continue
            coefs = np.asarray(line[1:], dtype='float32')
            if np.max(coefs) > embedding_max_value:
                embedding_max_value = np.max(coefs)
            if np.min(coefs) < embedding_min_value:
                embedding_min_value = np.min(coefs)
            embeddings_index[line[0]] = coefs
            word_index[line[0]]=i
            i=i+1
    embedword_matrix = np.zeros((word_num+1, 256))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedword_matrix[i] = embedding_vector
        else:#没有找到的词会被当做全0向量处理
            embedword_matrix[i] = np.random.uniform(low=embedding_min_value, high=embedding_max_value,
                                                             size=256)
    np.save('choose_paper/data/data_process_result/word_index.npy', word_index) 
    return embedword_matrix
    

def sample2sequence(SubSpace_train_pairs,SubSpace_dict):
    word_index = np.load('choose_paper/data/data_process_result/word_index.npy').item()
    index_list=list(word_index.keys())
    list_index_pos_first=[] 
    for each in SubSpace_train_pairs:
        temp_str=SubSpace_dict[each[0]]
        temp_list_word=temp_str.split(" ")
        temp_list_index=[]
        for i in temp_list_word:
            if i in index_list:
                temp_list_index.append(word_index[i])
        list_index_pos_first.append(temp_list_index)
    pos_index_pad_array_first = pad_sequences(list_index_pos_first, maxlen=150) 
    
    list_index_pos_sec=[]    
    for each in SubSpace_train_pairs:
        temp_str=SubSpace_dict[each[1]]
        temp_list_word=temp_str.split(" ")
        temp_list_index=[]
        for i in temp_list_word:
            if i in index_list:
                temp_list_index.append(word_index[i])
        list_index_pos_sec.append(temp_list_index)
    pos_index_pad_array_second = pad_sequences(list_index_pos_sec, maxlen=150) 
    
    return pos_index_pad_array_first,pos_index_pad_array_second






def test2sequence(firstId,secondId,SubSpace_dict):
    word_index = np.load('choose_paper/data/data_process_result/word_index.npy').item()
    index_list=list(word_index.keys())
    subspace_keys=SubSpace_dict.keys()
    if firstId in subspace_keys:
        temp_str1=SubSpace_dict[firstId]
        temp_list_word1=temp_str1.split(" ")
        temp_list_index1=[]    
        for i in temp_list_word1:
            if i in index_list:
                temp_list_index1.append(word_index[i])
        pad_array_first = pad_sequences([temp_list_index1], maxlen=150) 
    elif firstId not in subspace_keys:
        pad_array_first=np.array([[0]*150])
    if secondId in subspace_keys:
        temp_str2=SubSpace_dict[secondId]
        temp_list_word2=temp_str2.split(" ")
        temp_list_index2=[]    
        for i in temp_list_word2:
            if i in index_list:
                temp_list_index2.append(word_index[i])
        pad_array_second = pad_sequences([temp_list_index2], maxlen=150) 
    elif secondId not in subspace_keys:
        pad_array_second=np.array([[0]*150])
    return pad_array_first,pad_array_second

def rulesEmbbeding(firstId,secondId,subId):
    firstRule=referenceJaccard(firstId,secondId)
    secondRule=PaperId2KeywordsJaccard(firstId,secondId)
    thirdRule=ccsSimilarity(firstId,secondId)
    fourthRule=textJaccard(firstId,secondId,subId)
    FourRules=[firstRule,secondRule,thirdRule,fourthRule]
    return FourRules

##################################-------构建模型---------#############################################


    

class MyModel_rules():
    def __init__(self, batch_size=None, num_epochs=None, word_index=None,embedword_matrix=None, subId=None,
                 index_pad_array_first=None, index_pad_array_second=None, FourRules=None,y=None):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.word_index = word_index
        self.embedword_matrix=embedword_matrix
        self.subId=subId
        self.index_pad_array_first=index_pad_array_first
        self.index_pad_array_second=index_pad_array_first
        self.FourRules=FourRules
        self.y=y
        self.model = None
    def buildmodel(self):
        print('building model...')
        embedding_layer = Embedding(len(word_index) + 1,
                                           256,
                                           weights=[embedword_matrix],
                                           input_length=150, trainable=True)
        sequence_input1 = Input(shape=(150,), name="first_paper")
        sequence_input2 = Input(shape=(150,), name="second_paper")
        sequence_input3=Input(shape=(4,),name="rule")
        embedded_sequences1 = embedding_layer(sequence_input1)
        embedded_sequences2 = embedding_layer(sequence_input2)       
        LSTM_Left1 = LSTM(512, implementation=2, return_sequences=True, go_backwards=False)(embedded_sequences1)
        LSTM_Right1 = LSTM(512, implementation=2, return_sequences=True, go_backwards=True)(embedded_sequences1)
        concat1 = merge([LSTM_Left1,LSTM_Right1], mode='concat', concat_axis=-1)
        LSTM_Left2 = LSTM(512, implementation=2, return_sequences=True, go_backwards=False)(embedded_sequences2)
        LSTM_Right2 = LSTM(512, implementation=2, return_sequences=True, go_backwards=True)(embedded_sequences2)
        concat2 = merge([LSTM_Left2,LSTM_Right2], mode='concat', concat_axis=-1)
        z1 = Dense(512, activation='tanh')(concat1)
        z2 = Dense(512, activation='tanh')(concat2)
        z1_MaxPool = Lambda(lambda x: K.max(x, axis=1), output_shape=(512,))(z1)
        z2_MaxPool = Lambda(lambda x: K.max(x, axis=1), output_shape=(512,))(z2)
        concat=merge([z1_MaxPool,z2_MaxPool], mode='concat', concat_axis=-1)
        model_final = Dense(6, activation='relu')(concat)
        model_final=merge([model_final,sequence_input3],mode='concat',concat_axis=-1)
        model_final = Dropout(0.5)(model_final)
        model_final = Dense(2, activation='softmax')(model_final)
        self.model = Model(input=[sequence_input1, sequence_input2,sequence_input3],
                           outputs=model_final)
        adam = optimizers.adam(lr=0.0001)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=adam,
                           metrics=['accuracy'])
        print(self.model.summary())
    def trainmodel(self):
        self.buildmodel()
        checkpointer = ModelCheckpoint(filepath="model/"+str(self.subId)+"_model-{epoch:02d}.hdf5", period=1)
        y_train= np.asarray(self.y).astype('float32')
        self.model.fit([self.index_pad_array_first,self.index_pad_array_second,self.FourRules],y_train,
                    self.batch_size,self.num_epochs, verbose=1,
                           callbacks=[checkpointer])
        self.save_model()
    def predmodel(self,modelname,index_pad_array_first, index_pad_array_second,threeRules): 
        self.model = load_model(modelname)
        predlabel = self.model.predict([index_pad_array_first, index_pad_array_second, threeRules],
                                           batch_size=512, verbose=1)
        return predlabel
    def save_model(self):
        self.model.save("choose_paper/data/model/model" +str(self.subId)+ '.h5')


class MyModel_noRule():
    def __init__(self, batch_size=None, num_epochs=None, word_index=None,embedword_matrix=None, subId=None,
                 index_pad_array_first=None, index_pad_array_second=None,y=None):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.word_index = word_index
        self.embedword_matrix=embedword_matrix
        self.subId=subId
        self.index_pad_array_first=index_pad_array_first
        self.index_pad_array_second=index_pad_array_first
        self.y=y
        self.model = None
    def buildmodel(self):
        print('building model...')
        embedding_layer = Embedding(len(word_index) + 1,
                                           256,
                                           weights=[embedword_matrix],
                                           input_length=150, trainable=True)
        sequence_input1 = Input(shape=(150,), name="first_paper")
        sequence_input2 = Input(shape=(150,), name="second_paper")
        embedded_sequences1 = embedding_layer(sequence_input1)
        embedded_sequences2 = embedding_layer(sequence_input2)       
        LSTM_Left1 = LSTM(512, implementation=2, return_sequences=True, go_backwards=False)(embedded_sequences1)
        LSTM_Right1 = LSTM(512, implementation=2, return_sequences=True, go_backwards=True)(embedded_sequences1)
        concat1 = merge([LSTM_Left1,LSTM_Right1], mode='concat', concat_axis=-1)
        LSTM_Left2 = LSTM(512, implementation=2, return_sequences=True, go_backwards=False)(embedded_sequences2)
        LSTM_Right2 = LSTM(512, implementation=2, return_sequences=True, go_backwards=True)(embedded_sequences2)
        concat2 = merge([LSTM_Left2,LSTM_Right2], mode='concat', concat_axis=-1)
        z1 = Dense(512, activation='tanh')(concat1)
        z2 = Dense(512, activation='tanh')(concat2)
        z1_MaxPool = Lambda(lambda x: K.max(x, axis=1), output_shape=(512,))(z1)
        z2_MaxPool = Lambda(lambda x: K.max(x, axis=1), output_shape=(512,))(z2)
        concat=merge([z1_MaxPool,z2_MaxPool], mode='concat', concat_axis=-1)
        model_final = Dense(6, activation='relu')(concat)
        model_final = Dropout(0.5)(model_final)
        model_final = Dense(2, activation='softmax')(model_final)
        self.model = Model(input=[sequence_input1, sequence_input2],
                           outputs=model_final)
        adam = optimizers.adam(lr=0.0001)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=adam,
                           metrics=['accuracy'])
        print(self.model.summary())
    def trainmodel(self):
        self.buildmodel()
        checkpointer = ModelCheckpoint(filepath="model_/"+str(self.subId)+"_model-{epoch:02d}.hdf5", period=1)
        y_train= np.asarray(self.y).astype('float32')
        self.model.fit([self.index_pad_array_first,self.index_pad_array_second],y_train,
                    self.batch_size,self.num_epochs, verbose=1,
                           callbacks=[checkpointer])
        self.save_model()
    def predmodel(self,modelname,index_pad_array_first, index_pad_array_second): 
        self.model = load_model(modelname)
        predlabel = self.model.predict([index_pad_array_first, index_pad_array_second],
                                           batch_size=512, verbose=1)
        return predlabel
    def save_model(self):
        self.model.save("choose_paper/data/model_/model" +str(self.subId)+ '.h5')
  



if __name__ == '__main__':
    embedword_matrix=embeddingMatrix()
    word_index = np.load('choose_paper/data/data_process_result/word_index.npy').item()
    index_list=list(word_index.keys())

    SubSpace0_dict = np.load('choose_paper/data/data_process_result/SubSpace0_dict.npy').item()
    SubSpace1_dict = np.load('choose_paper/data/data_process_result/SubSpace1_dict.npy').item()
    SubSpace2_dict = np.load('choose_paper/data/data_process_result/SubSpace2_dict.npy').item()
    SubSpace3_dict = np.load('choose_paper/data/data_process_result/SubSpace3_dict.npy').item()  
    SubSpace4_dict = np.load('choose_paper/data/data_process_result/SubSpace4_dict.npy').item()
    
    

    SubSpace0_train_pairs=np.load('choose_paper/data/data_process_result/SubSpace0_train_pairs.npy')
    SubSpace0_train_pairs=SubSpace0_train_pairs.tolist()
    
    
    SubSpace1_train_pairs=np.load('choose_paper/data/data_process_result/SubSpace1_train_pairs.npy')
    SubSpace1_train_pairs=SubSpace1_train_pairs.tolist()
    
    SubSpace2_train_pairs=np.load('choose_paper/data/data_process_result/SubSpace2_train_pairs.npy')
    SubSpace2_train_pairs=SubSpace2_train_pairs.tolist()
   
    SubSpace3_train_pairs=np.load('choose_paper/data/data_process_result/SubSpace3_train_pairs.npy')
    SubSpace3_train_pairs=SubSpace3_train_pairs.tolist()
    
    SubSpace4_train_pairs=np.load('choose_paper/data/data_process_result/SubSpace4_train_pairs.npy')
    SubSpace4_train_pairs=SubSpace4_train_pairs.tolist()
    

##-----子空间0-------#############---模型训练------############################################################################################
    """
    index_pad_array0_first,index_pad_array0_second=sample2sequence(SubSpace0_train_pairs,SubSpace0_dict)
    FourRules0=[]
    for i in range(0,200):
        temp_list=rulesEmbbeding(SubSpace0_train_pairs[i][0],SubSpace0_train_pairs[i][1],0)
        FourRules0.append(temp_list)   
    FourRules0=np.array(FourRules0)
    y0=[]
    for i in range(0,200):
        if i%5==0:
            y0.append([1,0])
        elif SubSpace0_train_pairs[i][0]==SubSpace0_train_pairs[i][1]:
            y0.append([1,0])
        else:
            y0.append([0,1])
    
    model0=MyModel_rules(batch_size=128, num_epochs=config.NUM_EPOCHES, word_index=word_index,embedword_matrix=embedword_matrix, subId=0,
                 index_pad_array_first=index_pad_array0_first, index_pad_array_second=index_pad_array0_second,FourRules=FourRules0,y=y0)
    model0.trainmodel()
    
    
    model0_=MyModel_noRule(batch_size=128, num_epochs=config.NUM_EPOCHES, word_index=word_index,embedword_matrix=embedword_matrix, subId=0,
                 index_pad_array_first=index_pad_array0_first, index_pad_array_second=index_pad_array0_second,y=y0)
    model0_.trainmodel()
    """
##-----子空间1--------#################---模型训练------############################################################################################
    
    index_pad_array1_first,index_pad_array1_second=sample2sequence(SubSpace1_train_pairs,SubSpace1_dict)
    FourRules1=[]
    for i in range(0,200):
        temp_list=rulesEmbbeding(SubSpace1_train_pairs[i][0],SubSpace1_train_pairs[i][1],1)
        FourRules1.append(temp_list)   
    FourRules1=np.array(FourRules1)
    y1=[]
    for i in range(0,200):
        if i%5==0:
            y1.append([1,0])
        elif SubSpace1_train_pairs[i][0]==SubSpace1_train_pairs[i][1]:
            y1.append([1,0])
        else:
            y1.append([0,1])
    model1=MyModel_rules(batch_size=128, num_epochs=config.NUM_EPOCHES, word_index=word_index, embedword_matrix=embedword_matrix,subId=1,
                 index_pad_array_first=index_pad_array1_first, index_pad_array_second=index_pad_array1_second,FourRules=FourRules1,y=y1)
    model1.trainmodel()
    
    model1_=MyModel_noRule(batch_size=128, num_epochs=config.NUM_EPOCHES, word_index=word_index, embedword_matrix=embedword_matrix,subId=1,
                 index_pad_array_first=index_pad_array1_first, index_pad_array_second=index_pad_array1_second,y=y1)
    model1_.trainmodel()
    
##-----子空间2--------#################---模型训练------############################################################################################
    """
    index_pad_array2_first,index_pad_array2_second=sample2sequence(SubSpace2_train_pairs,SubSpace2_dict)
    FourRules2=[]
    for i in range(0,200):
        temp_list=rulesEmbbeding(SubSpace2_train_pairs[i][0],SubSpace2_train_pairs[i][1],2)
        FourRules2.append(temp_list)   
    FourRules2=np.array(FourRules2)
    y2=[]
    for i in range(0,200):
        if i%5==0:
            y2.append([1,0])
        elif SubSpace2_train_pairs[i][0]==SubSpace2_train_pairs[i][1]:
            y2.append([1,0])
        else:
            y2.append([0,1])
    model2=MyModel_rules(batch_size=128, num_epochs=config.NUM_EPOCHES, word_index=word_index,embedword_matrix=embedword_matrix, subId=2,
                 index_pad_array_first=index_pad_array2_first, index_pad_array_second=index_pad_array2_second,FourRules=FourRules2,y=y2)
    model2.trainmodel()

    model2_=MyModel_noRule(batch_size=128, num_epochs=config.NUM_EPOCHES, word_index=word_index,embedword_matrix=embedword_matrix, subId=2,
                 index_pad_array_first=index_pad_array2_first, index_pad_array_second=index_pad_array2_second,y=y2)
    model2_.trainmodel()    
##-----子空间3--------#################---模型训练------############################################################################################
    
    index_pad_array3_first,index_pad_array3_second=sample2sequence(SubSpace3_train_pairs,SubSpace3_dict)
    FourRules3=[]
    for i in range(0,200):
        temp_list=rulesEmbbeding(SubSpace3_train_pairs[i][0],SubSpace3_train_pairs[i][1],3)
        FourRules3.append(temp_list)   
    FourRules3=np.array(FourRules3)
    y3=[]
    for i in range(0,200):
        if i%5==0:
            y3.append([1,0])
        elif SubSpace3_train_pairs[i][0]==SubSpace3_train_pairs[i][1]:
            y3.append([1,0])
        else:
            y3.append([0,1])
    model3=MyModel_rules(batch_size=128, num_epochs=config.NUM_EPOCHES, word_index=word_index, embedword_matrix=embedword_matrix,subId=3,
                 index_pad_array_first=index_pad_array3_first, index_pad_array_second=index_pad_array3_second,FourRules=FourRules3,y=y3)
    model3.trainmodel()
    
    model3_=MyModel_noRule(batch_size=128, num_epochs=config.NUM_EPOCHES, word_index=word_index, embedword_matrix=embedword_matrix,subId=3,
                 index_pad_array_first=index_pad_array3_first, index_pad_array_second=index_pad_array3_second,y=y3)
    model3_.trainmodel()
  
##-----子空间4--------#################---模型训练------############################################################################################

    index_pad_array4_first,index_pad_array4_second=sample2sequence(SubSpace4_train_pairs,SubSpace4_dict)
    FourRules4=[]
    for i in range(0,200):
        temp_list=rulesEmbbeding(SubSpace4_train_pairs[i][0],SubSpace4_train_pairs[i][1],4)
        FourRules4.append(temp_list)   
    FourRules4=np.array(FourRules4)
    y4=[]
    for i in range(0,200):
        if i%5==0:
            y4.append([1,0])
        elif SubSpace4_train_pairs[i][0]==SubSpace4_train_pairs[i][1]:
            y4.append([1,0])
        else:
            y4.append([0,1])
    model4=MyModel_rules(batch_size=128, num_epochs=config.NUM_EPOCHES, word_index=word_index,embedword_matrix=embedword_matrix, subId=4,
                 index_pad_array_first=index_pad_array4_first, index_pad_array_second=index_pad_array4_second,FourRules=FourRules4,y=y4)
    model4.trainmodel()
    
    model4_=MyModel_noRule(batch_size=128, num_epochs=config.NUM_EPOCHES, word_index=word_index,embedword_matrix=embedword_matrix, subId=4,
                 index_pad_array_first=index_pad_array4_first, index_pad_array_second=index_pad_array4_second,y=y4)
    model4_.trainmodel()  
    """
#####------无规则

##-----子空间0-------#############---模型训练------############################################################################################
    """
    model0_=MyModel_noRule(batch_size=128, num_epochs=config.NUM_EPOCHES, word_index=word_index,embedword_matrix=embedword_matrix, subId=0,
                 index_pad_array_first=index_pad_array0_first, index_pad_array_second=index_pad_array0_second,y=y0)
    model0_.trainmodel()

##-----子空间1--------#################---模型训练------############################################################################################
    
    model1_=MyModel_noRule(batch_size=128, num_epochs=config.NUM_EPOCHES, word_index=word_index, embedword_matrix=embedword_matrix,subId=1,
                 index_pad_array_first=index_pad_array1_first, index_pad_array_second=index_pad_array1_second,y=y1)
    model1_.trainmodel()

##-----子空间2--------#################---模型训练------############################################################################################

    model2_=MyModel_noRule(batch_size=128, num_epochs=config.NUM_EPOCHES, word_index=word_index,embedword_matrix=embedword_matrix, subId=2,
                 index_pad_array_first=index_pad_array2_first, index_pad_array_second=index_pad_array2_second,y=y2)
    model2_.trainmodel()
    
##-----子空间3--------#################---模型训练------############################################################################################
 
    model3_=MyModel_noRule(batch_size=128, num_epochs=config.NUM_EPOCHES, word_index=word_index, embedword_matrix=embedword_matrix,subId=3,
                 index_pad_array_first=index_pad_array3_first, index_pad_array_second=index_pad_array3_second,y=y3)
    model3_.trainmodel()

##-----子空间4--------#################---模型训练------############################################################################################

    model4_=MyModel_noRule(batch_size=128, num_epochs=config.NUM_EPOCHES, word_index=word_index,embedword_matrix=embedword_matrix, subId=4,
                 index_pad_array_first=index_pad_array4_first, index_pad_array_second=index_pad_array4_second,y=y4)
    model4_.trainmodel()
    """