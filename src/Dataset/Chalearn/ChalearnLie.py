'''
Created on Apr 9, 2018

@author: hshi
'''
import os
import numpy as np
import h5py
from Dataset.Dataset import Dataset
class ChalearnLie(Dataset):
    '''
    classdocs
    '''


    def __init__(self, mConfig):
        '''
        Constructor
        '''
        self.dataPath = mConfig.dataPath
        
        self.dataPath_train = os.path.join(self.dataPath, 'train')
        self.dataPath_test = os.path.join(self.dataPath, 'test')
        

        
    def getData_vaeReconstruction_lie(self, mConfig):
        
        def generateMask(segLenList, segLen_max):
            maskList = list()
            for i in range(len(segLenList)):
                maskList.append(np.zeros(segLen_max))
                maskList[i][segLenList[i]-1] = 1
            
            return maskList
        

        
        minimumLen = 1
        if mConfig.doHmm:
            minimumLen = 1 + mConfig.stateNumPerClass
            
            
    
        
        sampleList_train  = list() 
        frameLabelList_train  = list() 
        segLenList_train  = list()
        
        sampleList_test  = list() 
        frameLabelList_test  = list() 
        segLenList_test = list()
        
        reconstructionSampleList_train  = list() 
        reconstructionSampleList_test  = list()
            
            
            
        sampleNames_train = os.listdir(self.dataPath_train)
        
        for i in range(len(sampleNames_train)):
            currentFile = h5py.File(os.path.join(self.dataPath_train, sampleNames_train[i]))
            currentSample = currentFile['data']['joint_locations'].value
            currentRecon = currentFile['data']['lieAl'].value
            
            
            sampleList_train.append(currentSample.reshape([-1, 60]))
            reconstructionSampleList_train.append(currentRecon)
            currentLabel = np.zeros(currentSample.shape[0])
            currentLabel[:] = currentFile['data']['action'].value[0,0] - 1
            frameLabelList_train.append(currentLabel)
            segLenList_train.append(currentSample.shape[0])
            currentFile.close()
        
        sampleNames_test = os.listdir(self.dataPath_test)
        
        for i in range(len(sampleNames_test)):
            currentFile = h5py.File(os.path.join(self.dataPath_test, sampleNames_test[i]))
            currentSample = currentFile['data']['joint_locations'].value
            currentRecon = currentFile['data']['lieAl'].value
            
            
            sampleList_test.append(currentSample.reshape([-1, 60]))
            reconstructionSampleList_test.append(currentRecon)
            currentLabel = np.zeros(currentSample.shape[0])
            currentLabel[:] = currentFile['data']['action'].value[0,0] - 1
            frameLabelList_test.append(currentLabel)
            segLenList_test.append(currentSample.shape[0])                          
            currentFile.close()

        mConfig.reconstructionTargetDim = currentRecon.shape[-1]
        
        if mConfig.doNormalization is True:
            priorScalar = self.fitNormalizer(mConfig.normalizationStyle, \
                                             sampleList_train, segLenList_train, sampleList_test, segLenList_test)
   
            priorScalar1 = self.fitNormalizer(mConfig.normalizationStyle, \
                                              reconstructionSampleList_train, segLenList_train, \
                                              reconstructionSampleList_test, segLenList_test)
        
        
            sampleList_train, sampleList_test, _ = self.normalize(priorScalar, sampleList_train, sampleList_test)
            reconstructionSampleList_train, reconstructionSampleList_test, reconstructionSampleList_valid = self.normalize(priorScalar1, reconstructionSampleList_train, reconstructionSampleList_test)
            
            
        else:
            pass
            
        if mConfig.doHmm is True:
            mConfig.classNum = mConfig.classNum * mConfig.stateNumPerClass
            frameLabelList_train = self.frameLabelList2FrameStateList(frameLabelList_train, mConfig.stateNumPerClass)
            frameLabelList_test = self.frameLabelList2FrameStateList(frameLabelList_test, mConfig.stateNumPerClass)

        else:
            pass
        
        

        segLenMat_train = np.array(segLenList_train)
        segLenMat_test = np.array(segLenList_test)
        
  
        segLen_max = np.max([np.max(segLenMat_train), np.max(segLenMat_test)])
    
        sampleList_train_padded, frameLabelList_train_padded, sampleList_test_padded, frameLabelList_test_padded, sampleList_valid_padded, frameLabelList_valid_padded = \
            self.padding(segLen_max, sampleList_train, frameLabelList_train, sampleList_test, frameLabelList_test)
            
            
        reconstructionSampleList_train_padded = self.paddingData(reconstructionSampleList_train, segLen_max)
        reconstructionSampleList_test_padded = self.paddingData(reconstructionSampleList_test, segLen_max)    


        
        mConfig.reconstructionTargetDim = reconstructionSampleList_train[0].shape[-1]
        
        batchSource_train = list()
        batchSource_train.append(sampleList_train_padded)
        batchSource_train.append(frameLabelList_train_padded)
        batchSource_train.append(segLenList_train)
        batchSource_train.append(reconstructionSampleList_train_padded)
                
        batchSource_test = list()
        batchSource_test.append(sampleList_test_padded)
        batchSource_test.append(frameLabelList_test_padded)
        batchSource_test.append(segLenList_test)
        batchSource_test.append(reconstructionSampleList_test_padded)
            
        sampleDim = sampleList_train_padded[0].shape[-1]
        

        
        return batchSource_train, batchSource_test, segLen_max, sampleDim, \
            frameLabelList_train, frameLabelList_test
        
        
        
    def getData(self, mConfig):
        
        
        
        minimumLen = 1
        if mConfig.doHmm:
            minimumLen = 1 + mConfig.stateNumPerClass
            
            
        sampleList_train  = list() 
        frameLabelList_train  = list() 
        segLenList_train  = list()
        
        sampleList_test  = list() 
        frameLabelList_test  = list() 
        segLenList_test = list()
        
            
            
            
        sampleNames_train = os.listdir(self.dataPath_train)
        
        for i in range(len(sampleNames_train)):
            currentFile = h5py.File(os.path.join(self.dataPath_train, sampleNames_train[i]))
            currentSample = currentFile['data']['joint_locations'].value
            currentRecon = currentFile['data']['lieAl'].value
            
            
            sampleList_train.append(currentRecon)
            currentLabel = np.zeros(currentSample.shape[0])
            currentLabel[:] = currentFile['data']['action'].value[0,0] - 1
            frameLabelList_train.append(currentLabel)
            segLenList_train.append(currentSample.shape[0])
            currentFile.close()
        
        sampleNames_test = os.listdir(self.dataPath_test)
        
        for i in range(len(sampleNames_test)):
            currentFile = h5py.File(os.path.join(self.dataPath_test, sampleNames_test[i]))
            currentSample = currentFile['data']['joint_locations'].value
            currentRecon = currentFile['data']['lieAl'].value
            
            
            sampleList_test.append(currentRecon)
            currentLabel = np.zeros(currentSample.shape[0])
            currentLabel[:] = currentFile['data']['action'].value[0,0] - 1
            frameLabelList_test.append(currentLabel)
            segLenList_test.append(currentSample.shape[0])                          
            currentFile.close()

        mConfig.reconstructionTargetDim = currentRecon.shape[-1]
        
        


        if mConfig.doNormalization is True:
            priorScalar = self.fitNormalizer(mConfig.normalizationStyle, \
                                             sampleList_train, segLenList_train, sampleList_test, segLenList_test)
   

        
        
            sampleList_train, sampleList_test, sampleList_valid = self.normalize(priorScalar, sampleList_train, sampleList_test)
            
            
        else:
            pass
            
        if mConfig.doHmm is True:
            mConfig.classNum = mConfig.classNum * mConfig.stateNumPerClass
            frameLabelList_train = self.frameLabelList2FrameStateList(frameLabelList_train, mConfig.stateNumPerClass)
            frameLabelList_test = self.frameLabelList2FrameStateList(frameLabelList_test, mConfig.stateNumPerClass)
            
            

        
        

        segLenMat_train = np.array(segLenList_train)
        segLenMat_test = np.array(segLenList_test)

        segLen_max = np.max([np.max(segLenMat_train), np.max(segLenMat_test)])
    
        sampleList_train_padded, frameLabelList_train_padded, sampleList_test_padded, frameLabelList_test_padded, sampleList_valid_padded, frameLabelList_valid_padded = \
            self.padding(segLen_max, sampleList_train, frameLabelList_train, sampleList_test, frameLabelList_test)
            

        
        batchSource_train = list()
        batchSource_train.append(sampleList_train_padded)
        batchSource_train.append(frameLabelList_train_padded)
        batchSource_train.append(segLenList_train)
                
        batchSource_test = list()
        batchSource_test.append(sampleList_test_padded)
        batchSource_test.append(frameLabelList_test_padded)
        batchSource_test.append(segLenList_test)
            
        sampleDim = sampleList_train_padded[0].shape[-1]
        

    
        
        return batchSource_train, batchSource_test, segLen_max, sampleDim, \
            frameLabelList_train, frameLabelList_test, 
        
        
