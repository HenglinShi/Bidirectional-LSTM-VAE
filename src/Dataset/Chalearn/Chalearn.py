'''
Created on Jan 9, 2018

@author: hshi
'''

import os
import csv
import numpy as np
from Dataset.Dataset import Dataset
class ChaLearn(Dataset):
    '''
    classdocs
    '''


    def __init__(self, dataDir_root, minframeNum):
        '''
        Constructor
        '''
        self.dataDir_root = dataDir_root
        self.mean = None
        self.priorScalar = None

        
        
        self.data_train = None
        self.data_valid = None
        self.data_test = None
        
        self.sampleVec_train = None
        self.sampleVec_valid = None
        self.sampleVec_test = None
        # preloading the datasets
        
        
        self.dataDir_train = os.path.join(self.dataDir_root, 'train')
        self.dataDir_valid = os.path.join(self.dataDir_root, 'valid')
        self.dataDir_test = os.path.join(self.dataDir_root, 'test')
        
        
        self.data_train_obj = ChaLearn2014Subset(self.dataDir_train, minframeNum)
        self.data_valid_obj = ChaLearn2014Subset(self.dataDir_valid, minframeNum)
        self.data_test_obj = ChaLearn2014Subset(self.dataDir_test, minframeNum)
        
        
        
    def splitData(self, mConfig):
        self.data_train = self.data_train_obj.dataList
        self.data_test = self.data_test_obj.dataList
        if mConfig.validation is True:
            self.data_valid = None
        else:
            self.data_valid = self.data_valid_obj.dataList
        
        return self.data_train, self.data_test, self.data_valid

class ChaLearn2014Subset(object):
    
    def __init__(self, dataDir, minFrameNumPerSequence=2, centralized=False):
        
        self.dataDir = dataDir
        self.centralized = centralized

        
        self.minFrameNumPerSequence=minFrameNumPerSequence
        self.loadData()
        self.sampleNum = len(self.dataList)
        

            
    def loadData(self):

        
        dataNames = os.listdir(self.dataDir)
        
        self.dataList = list()
        
        dataNum = len(dataNames)
        dataNames.sort()
        for i in range(dataNum):
            
            currentDataPath = os.path.join(self.dataDir, dataNames[i])

            currentLabelPath = os.path.join(currentDataPath, dataNames[i] + '_labels.csv')
            currentSamplePath = os.path.join(currentDataPath, dataNames[i] + '_skeleton.csv')
            
            currentSample = self.loadOneSample(currentSamplePath)
            currentSequenceLabels = self.loadOneLabel(currentLabelPath)
            #shutil.rmtree(currentDataPath[:-4])
            
            for segIte in range(len(currentSequenceLabels)):
                
                currentSegSample = currentSample[currentSequenceLabels[segIte][1] - 1 : currentSequenceLabels[segIte][2], :]
                

                self.dataList.append({'sample': currentSegSample, \
                                      'label': currentSequenceLabels[segIte][0]})
                    

            


    def loadOneLabel(self, labelFilePath):
        tmpLabel=[]
        with open(labelFilePath, 'rb') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                tmpLabel.append(map(int,row))    
            del filereader
        return tmpLabel
        
    
                
    def loadOneSample(self, sampleFilePath):
            
        currentSample = []
            
        with open(sampleFilePath, 'rb') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                currentSample.append(map(float, row))
            del filereader
                
        return np.array(currentSample)    
    

    
    
    