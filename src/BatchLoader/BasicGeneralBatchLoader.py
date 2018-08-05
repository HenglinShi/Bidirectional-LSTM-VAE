'''
Created on Feb 21, 2018

@author: hshi
'''
import numpy as np
import random as rd
class BasicGeneralBatchLoader(object):
    '''
    classdocs
    '''

    def __init__(self, batchSize, dataList):
        '''
        Constructor
        '''
        self.dataList = dataList
        self.batchSize = batchSize
        self.sampleNum = len(self.dataList[0])
        self.cursor = 0
        self.reset()
        
        
    def reset(self):
        self.cursor = 0
        
        ind_ = np.linspace(0, self.sampleNum - 1, self.sampleNum).astype('int')
        rd.shuffle(ind_)
        
        newDataList = list()
        
        for _ in range(len(self.dataList)):
            newDataList.append(list())
        
        
        for i in range(self.sampleNum):
            
            for dataIte in range(len(self.dataList)):
                newDataList[dataIte].append(self.dataList[dataIte][ind_[i]])
                
        self.dataList = newDataList
        
    def getNextBatch(self):
        
        outDataList = list()
        
        for _ in range(len(self.dataList)):
            outDataList.append(list())
        
        for _ in range(self.batchSize):
            if self.cursor == self.sampleNum:
                self.cursor = 0
                self.reset()
            
            
            for dataIte in range(len(self.dataList)):
                outDataList[dataIte].append(self.dataList[dataIte][self.cursor])
            
            self.cursor += 1
            
        return outDataList

        
    def getBatchNumPerEpoch(self):
        
        return np.ceil(self.sampleNum * 1.0 / self.batchSize).astype('int') 