'''
Created on Mar 17, 2018

@author: hshi
'''
import os
import numpy as np
from Dataset.Dataset import Dataset

class MsrAction3d(Dataset):
    '''
    classdocs
    '''


    def __init__(self, dataDir, subject_train = None, subject_test = None):
        '''
        Constructor
        '''
        Dataset.__init__(self)
        self.dataDir = dataDir
        #self.crossSubject = mConfig.crossSubject
        self.loadData()


        

    
        
    def loadData(self):
        self.dataList = list()
        sampleNames = os.listdir(self.dataDir)
        
        self.sampleNum = len(sampleNames)
        
        for sampleIte in range(self.sampleNum):
            currentSamplePath = os.path.join(self.dataDir, sampleNames[sampleIte])
            
            #with open(currentSamplePath, 'r') as f:
            currentSample = np.loadtxt(currentSamplePath)[:,:3]
            currentSample = currentSample.reshape([-1, 20 * 3])
            
            #self.sampleList.append(currentSample)
            
            actionLabel, subjectId, repetation, _ = sampleNames[sampleIte].split('.')[0].split('_')
            actionLabel = int(actionLabel[1:])
            subjectId = subjectId[1:]
            repetation = repetation[1:]
            
            self.dataList.append({'sample': currentSample,\
                                  'label': actionLabel,\
                                  'subjectId': subjectId,\
                                  'repetation': repetation})
            
            
