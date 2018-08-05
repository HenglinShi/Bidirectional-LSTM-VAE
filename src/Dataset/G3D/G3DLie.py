'''
Created on Apr 10, 2018

@author: hshi
'''
from Dataset.Dataset import Dataset
import os
import numpy as np
import h5py

class G3DLie(Dataset):
    '''
    classdocs
    '''


    def __init__(self, mConfig):
        '''
        Constructor
        '''
        self.dataPath = mConfig.dataPath
        
        self.dataList = list()
        
        sampleNames = os.listdir(self.dataPath)
        
        for i in range(len(sampleNames)):
            currentFile = h5py.File(os.path.join(self.dataPath, sampleNames[i]))
            currentRecon = currentFile['currentData']['lieAl'].value
            currentLabel = int(currentFile['currentData']['label'].value[0,0])
            currentSubject = str(int(currentFile['currentData']['subjectId'].value[0,0]))
            print (str(currentLabel) + '          ' + str(currentSubject))
            currentFile.close()
            
            
            self.dataList.append({'sample': currentRecon,\
                                  'label': currentLabel,\
                                  'subjectId': currentSubject})
            
            
    def getSample(self, data, minFrame, mConfig):
        #usedJoints, dimPerJoint = 3, dataType = '3dCoordinate', centralize = False, centerJointInd = 1):
        

        outSampleList = list()
        outFrameLabelList = list()
        segLenList = list()
        

    
        for i in range(len(data)):
            currentSample = data[i]['sample']
            

            
            currentLabel = np.zeros(len(currentSample))
            currentLabel[:] = data[i]['label'] - 1
            
            currentSample, currentLabel = self.filteringInvalidFrames(currentSample, currentLabel)
            
            if currentSample.shape[0] >= minFrame:
                outSampleList.append(currentSample)
                outFrameLabelList.append(currentLabel)
                segLenList.append(currentSample.shape[0])
            else:
                print (i)
            
            
        return outSampleList, outFrameLabelList, segLenList
    
    