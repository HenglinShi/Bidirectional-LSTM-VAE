'''
Created on Feb 21, 2018

@author: hshi
'''
import numpy as np
def extractRelativeJointFeatureFromDataList_spatial_temporal(sampleList, frameLabelList, usedJoints=np.linspace(0, 24, 25).astype('int')):
        
        
    outSampleList = list()
    outFrameLabelList = list()
    segLenList = list()
            
    jointNum = len(usedJoints)
    #selectedColumns = np.concatenate((usedJoints * 3, usedJoints * 3 + 1, usedJoints * 3 + 2))
    #selectedColumns.sort()
    #featureMat_all = np.zeros([1, ((jointNum * (jointNum)/2)*3) + ((jointNum ** 2) * 3)])
            
    for i in range(len(sampleList)):
        currentSample = sampleList[i]#[:, selectedColumns]
                
        currentSpatialFeature = extractRelativeJointFeatureFromMat_spatial(currentSample, jointNum)
        currentTemporalFeature = extractRelativeJointFeatureFromMat_temporal(currentSample, jointNum)
                
        currentFeature = np.concatenate((currentSpatialFeature[1: , :], currentTemporalFeature), axis = 1)
            
        outSampleList.append(currentFeature)
        outFrameLabelList.append(frameLabelList[i][1:])
        segLenList.append(currentFeature.shape[0])
            
            
    return outSampleList, outFrameLabelList, segLenList

def extractRelativeJointFeatureFromMat_spatial(sampleMat, jointNum):
    feature = np.zeros((sampleMat.shape[0], (jointNum * (jointNum - 1)/2)*3))
        
    featureColumnIte = 0
    for jointIte1 in range(jointNum - 1):
        for jointIte2 in range(jointIte1 + 1, jointNum):
            feature[:,featureColumnIte * 3 : (featureColumnIte + 1) * 3] = \
                sampleMat[:, jointIte1 * 3 : (jointIte1 + 1) * 3] - sampleMat[:, jointIte2 * 3 : (jointIte2 + 1) * 3]
            featureColumnIte += 1
        
    return feature
    
    
    
def extractRelativeJointFeatureFromMat_temporal(sampleMat, jointNum):
        
    feature = np.zeros((sampleMat.shape[0] - 1, (jointNum ** 2) * 3))
        
    featureColumnIte = 0
        
    for jointIte1 in range(jointNum):
        for jointIte2 in range(jointNum):
            feature[:, featureColumnIte * 3 : (featureColumnIte + 1) * 3] = \
                sampleMat[1:, jointIte1 * 3 : (jointIte1 + 1) * 3] - sampleMat[0:-1, jointIte2 * 3 : (jointIte2 + 1) * 3]
                
            featureColumnIte += 1
                
    return feature
def list2Mat_fast(inList, sequenceLengthsList):
    sequenceLengths = np.array(sequenceLengthsList)
    totalLength = sequenceLengths.sum()
    outMat = np.zeros([totalLength, inList[0].shape[-1]])
    beg = 0
    end = 0
    
    for i in range(len(inList)):
        beg = end
        end = beg + sequenceLengths[i]
        outMat[beg:end, :] = inList[i]
        
    return outMat[1:]
def frameLabelList2FrameStateList(frameLabelList, stateNumPerClass):
    
    frameStateList = list()
    
    for i in range(len(frameLabelList)):
        currentFrameLabels = frameLabelList[i]
        currentFrameStates = currentFrameLabels * stateNumPerClass + np.round(np.linspace(-0.5, stateNumPerClass - 0.51, currentFrameLabels.shape[0])).astype('int')
        frameStateList.append(currentFrameStates)
    
    return frameStateList

def basicNormalization(sampleList, prior):
    
    for i in range(len(sampleList)):
        sampleList[i] = prior.transform(sampleList[i])

    return sampleList


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

