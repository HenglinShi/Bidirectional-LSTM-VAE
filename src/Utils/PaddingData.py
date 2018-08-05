'''
Created on Feb 21, 2018

@author: hshi
'''
import numpy as np
def paddingData(sampleList, segLen_max):
        
    paddedDataList = list()
    sampleNum = len(sampleList)
    dataShape = list(sampleList[0].shape)
    dataShape[0] = segLen_max
    
    for sampleIte in range(sampleNum):
        currentPaddedData = np.zeros(dataShape)
        currentPaddedData[:sampleList[sampleIte].shape[0],...] = sampleList[sampleIte]
        paddedDataList.append(currentPaddedData)
        
    return paddedDataList