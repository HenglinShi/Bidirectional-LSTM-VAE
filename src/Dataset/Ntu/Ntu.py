'''
Created on Feb 20, 2018

@author: hshi
'''
import cPickle as pickle
import numpy as np
import random as rd
from Sample import Sample
import os
from Dataset.Dataset import Dataset
import scipy.io as sio
class Ntu(Dataset):
    '''
    classdocs
    '''


    def __init__(self, dataPath, mConfig, subject_train = None, subject_test = None, dataRegen = False):
        '''
        Constructor
        '''
        self.sampleList = list()
        self.labelList = list()
        self.dataList = None
        self.dataPath = dataPath
        if dataRegen:
            #self.rootDir = dataPath
            self.sampleNum = None
            self.sampleNames= None
            self.rawDataPath = mConfig.rawDataPath
            self.processFromRawData(mConfig)
            

        #self.dataPath = dataPath
            
        mFile = open(self.dataPath, 'rb')
        data = pickle.load(mFile)
        mFile.close()
        self.data = data['sample']
            
        self.stackingFrames()
        self.dataList = self.data
            

            
            
            
    def processFromRawData(self, mConfig):
        
        self.dataList = list()
        
        sampleNames = os.listdir(self.rawDataPath)
        self.invalidList = list()
        for i in range(len(sampleNames)):
            currentSamplePath = os.path.join(self.rawDataPath, sampleNames[i])
            currentSample = Sample(currentSamplePath)            
            
            frameList = list()
            
        
        
        
            print ('processing ' + str(i))
            for frameIte in range(currentSample.frameNum):
        
                currentFrame = dict()
                if len(currentSample.frameList[frameIte].getBodyIds()) == 0:
                    print currentSamplePath
                    
                for bodyId in currentSample.frameList[frameIte].getBodyIds():
                    currentFrame[bodyId] = currentSample.frameList[frameIte].getData(bodyId, 'all')
                    
            
                frameList.append(currentFrame)
            
            
            tmpname = sampleNames[i].split('.')[0]
            self.dataList.append({'frameList':frameList,
                               'setup': tmpname[1:4],
                               'camera': tmpname[5:8],
                               'performer':tmpname[9:12],
                               'replication':tmpname[13:16],
                               'action': tmpname[17:20]})
    
        
        
        
        
        
        #self.dataPath = self.dataPath.split('.')[0] + '_1' + '.pkl'
        while os.path.exists(self.dataPath):
            self.dataPath = self.dataPath.split('.')[0] + '_1' + '.pkl'
        f = open(self.dataPath,'wb')
        pickle.dump({"sample": self.dataList},f)
        f.close()    
        
        
        
        
    def stackingFrames(self):
        
        
        def getKeys(frameList):
            keys = list()
            
            for i in range(len(frameList)):
                currentKeys = frameList[i].keys()
                
                for j in range(len(currentKeys)):
                    if currentKeys[j] not in keys:
                        keys.append(currentKeys[j])
                        
            return keys
        
        i = 0
        while (self.data[0]['frameList'][i].keys() == []):
            i = i + 1
        
        sampleDim = self.data[0]['frameList'][i][self.data[0]['frameList'][i].keys()[0]].shape[0]
        

        
        
        for i in range(len(self.data)):
            
            frameNum = len(self.data[i]['frameList'])
            subjectKeys = getKeys(self.data[i]['frameList'])
            currentSample = dict()
            
            if len(subjectKeys) > 2:
                print ('found frames with more than two subjects')
            for currentSubjectKey in subjectKeys:
                currentSample[currentSubjectKey] = np.zeros([frameNum, sampleDim])
            
            for frameIte in range(frameNum):
                
                subjectKeys_currentFrame = self.data[i]['frameList'][frameIte].keys()
                
                subjectKeys_pastFrame = self.data[i]['frameList'][max(0, frameIte-1)].keys()
                
                if subjectKeys_currentFrame != subjectKeys_pastFrame:
                    print ('subject changes')
                    
                if subjectKeys_currentFrame != []:
                    
                    for subjectKeyIte_currentFrame in subjectKeys_currentFrame:
                        currentSample[subjectKeyIte_currentFrame][frameIte, :] = self.data[i]['frameList'][frameIte][subjectKeyIte_currentFrame]
                
                else:
                    print('frame with no subjects')
            
            
            self.data[i]['sample'] = currentSample
            
    def crossSubjectSplitting(self, subjectList):
        subjectNum = len(subjectList)
        
        
        subjectInd = np.linspace(0, subjectNum - 1, subjectNum).astype('int')
        
        subjectInd_train = subjectInd[0:np.ceil(subjectNum/2).astype('int')]
        subjectInd_test = subjectInd[np.ceil(subjectNum/2).astype('int'):]
        
        rd.shuffle(subjectList)
        subjectList_train = list()
        subjectList_test = list()
        
        for i in range(len(subjectInd_train)):
            subjectList_train.append(subjectList[subjectInd_train[i]])

        for i in range(len(subjectInd_test)):
            subjectList_test.append(subjectList[subjectInd_test[i]])
            
        return subjectList_train, subjectList_test
        

    def getDataBySubjects(self, data, selectedSubjects):
        
        selectedData = list()
        
        for i in range(len(data)):
            currentSubject = data[i]['performer']
            
            if currentSubject in selectedSubjects:
                selectedData.append(data[i])
        
         
           
        return selectedData
        
    def getSubjects(self, data):
        
        
        subjectList = list()
        
        
        for i in range(len(data)):
            currentSubject = data[i]['performer']
            if currentSubject not in subjectList:
                subjectList.append(currentSubject)
                
        return subjectList
    
    
        
    
    def getData_longest(self, data, minFrame, dataType = '3dCoordinate rotation 2dCoordinate depthCoordinate', usedJoints=np.linspace(0, 24, 25).astype('int')):
        
        if dataType == '3dCoordinate rotation 2dCoordinate depthCoordinate':
            selectedColumns = np.concatenate((usedJoints * 12, usedJoints * 12 + 1, usedJoints * 12 + 2, \
                                              usedJoints * 12 + 3, usedJoints * 12 + 4, usedJoints * 12 + 5, usedJoints * 12 + 6, \
                                              usedJoints * 12 + 7, usedJoints * 12 + 8, \
                                              usedJoints * 12 + 9, usedJoints * 12 + 10))
        
        elif dataType == '3dCoordinate rotation 2dCoordinate':
            selectedColumns = np.concatenate((usedJoints * 12, usedJoints * 12 + 1, usedJoints * 12 + 2, \
                                              usedJoints * 12 + 3, usedJoints * 12 + 4, usedJoints * 12 + 5, usedJoints * 12 + 6, \
                                              usedJoints * 12 + 7, usedJoints * 12 + 8,))
        
        
        selectedColumns.sort()
        outSample = list()
        outLabel = list()
        outSegLen = list()
        
        
        
        for i in range(len(data)):
            keys = data[i]['sample'].keys()
            if len(keys) != 0:
                validLengths = np.zeros(len(keys)).astype('int')
                
                for keyIte in range(len(keys)):
                    currentSample = data[i]['sample'][keys[keyIte]][:,selectedColumns]
                    currentSample = np.sum(np.abs(currentSample), 1)
                    validFrameInd = np.where(currentSample > 0)[0]
                    validLengths[keyIte] = len(validFrameInd)
                    
                    
                longestInd = validLengths.argmax()
                currentSample = data[i]['sample'][keys[longestInd]][:,selectedColumns]
                currentLabel = np.zeros(len(currentSample))
                currentLabel[:] = int(data[i]['action']) - 1
                currentSample, currentLabel = self.filteringInvalidFrames(currentSample, currentLabel)
                
                if len(currentSample) > minFrame:
                    outSample.append(currentSample)
                    outLabel.append(currentLabel)
                    outSegLen.append(len(currentSample))
    
            else:
                #remember invalid files
                pass
        return outSample, outLabel, outSegLen
    
    

    def getData_longest_2(self, data, minFrame, dataType = '3dCoordinate rotation 2dCoordinate depthCoordinate', usedJoints=np.linspace(0, 24, 25).astype('int')):
        
        if dataType == '3dCoordinate rotation 2dCoordinate depthCoordinate':
            selectedColumns = np.concatenate((usedJoints * 12, usedJoints * 12 + 1, usedJoints * 12 + 2, \
                                              usedJoints * 12 + 3, usedJoints * 12 + 4, usedJoints * 12 + 5, usedJoints * 12 + 6, \
                                              usedJoints * 12 + 7, usedJoints * 12 + 8, \
                                              usedJoints * 12 + 9, usedJoints * 12 + 10))
        
        elif dataType == '3dCoordinate rotation 2dCoordinate':
            selectedColumns = np.concatenate((usedJoints * 12, usedJoints * 12 + 1, usedJoints * 12 + 2, \
                                              usedJoints * 12 + 3, usedJoints * 12 + 4, usedJoints * 12 + 5, usedJoints * 12 + 6, \
                                              usedJoints * 12 + 7, usedJoints * 12 + 8,))
        
        
        selectedColumns.sort()
        outSample = list()
        outLabel = list()
        outSegLen = list()
        
        sampleDim = len(selectedColumns)
        
        for i in range(len(data)):
            keys = data[i]['sample'].keys()
            if len(keys) != 0:
                
                frameNum = data[i]['sample'][keys[0]].shape[0]
                
                validLengths = np.zeros(len(keys)).astype('int')
                
                # Find the longest one
                validFrameInds = dict()
                for keyIte in range(len(keys)):
                    currentSample = data[i]['sample'][keys[keyIte]][:,selectedColumns]
                    currentSample = np.sum(np.abs(currentSample), 1)
                    currentValidFrameInd = np.where(currentSample > 0)[0]
                    
                    validFrameInds[keyIte] = currentValidFrameInd
                    validLengths[keyIte] = len(currentValidFrameInd)
                    
                longestInd = validLengths.argmax()
                # longestValidFrameInds = validFrameInds[longestInd]
                # Then finding the altenatetive one
                # What is the strategy
                # 1st find the max over lapping with the longest one
                
                tmpSample = np.zeros((frameNum, 2 * sampleDim))
                tmpSample[:, :sampleDim] = data[i]['sample'][keys[longestInd]][:,selectedColumns]
                
                if len(keys) > 1:
                    validLengths[longestInd] = 0
                    secondLongestInd = validLengths.argmax()
                    tmpSample[:, sampleDim:] = data[i]['sample'][keys[secondLongestInd]][:,selectedColumns]

                currentLabel = np.zeros(len(tmpSample))
                currentLabel[:] = int(data[i]['action']) - 1
                currentSample, currentLabel = self.filteringInvalidFrames(tmpSample, currentLabel)
                
                if len(currentSample) > minFrame:
                    outSample.append(currentSample)
                    outLabel.append(currentLabel)
                    outSegLen.append(len(currentSample))
    
            else:
                #remember invalid files
                pass
        return outSample, outLabel, outSegLen
    
   

    def getSample_activest2(self, data, minFrame, mConfig):
        
        usedJoints = mConfig.selectedJoint
        dimPerJoint = mConfig.dimPerJoint
        dataType = mConfig.dataType
        centralize = mConfig.centralize
        if centralize is True:
            centerJointInd = mConfig.centerJointInd
        selectedColumns = self.getDataColumnInd(usedJoints, dimPerJoint, dataType)
        selectedColumns.sort()
        outSampleList = list()
        outFrameLabelList = list()
        segLenList = list()
        
        jointNum = len(usedJoints)
        
        sampleDim = len(selectedColumns)
        
        for i in range(len(data)):
            keys = data[i]['sample'].keys()
            if len(keys) != 0:
                
                frameNum = data[i]['sample'][keys[0]].shape[0]
                
                activeScores = np.zeros(len(keys))
                # Find the longest one
                for keyIte in range(len(keys)):
                    currentSample = data[i]['sample'][keys[keyIte]][:,selectedColumns]
                    activeScores[keyIte] = self.getActivityScore(currentSample)
                    
                
                highestSampleKeyInd = activeScores.argmax()
                
                currentSample = np.zeros((frameNum, 2 * sampleDim))
                
                tmpSample = data[i]['sample'][keys[highestSampleKeyInd]][:,selectedColumns]
                if centralize is True:
                    tmpSample = tmpSample - np.matlib.repmat(tmpSample[:,centerJointInd * 3 : (centerJointInd + 1) * 3], 1, jointNum)
                    
                currentSample[:,:sampleDim] = tmpSample
                
                if len(keys) > 1:
                    activeScores[highestSampleKeyInd] = 0
                    secondHighestKeyInd = activeScores.argmax()
                    tmpSample = data[i]['sample'][keys[secondHighestKeyInd]][:,selectedColumns]
                    if centralize is True:
                        tmpSample = tmpSample - np.matlib.repmat(tmpSample[:,centerJointInd * 3 : (centerJointInd + 1) * 3], 1, jointNum)
                    currentSample[:,sampleDim:] = tmpSample
                    
                    
                currentLabel = np.zeros(len(currentSample))
                currentLabel[:] = int(data[i]['action']) - 1
                currentSample, currentLabel = self.filteringInvalidFrames(currentSample, currentLabel)
                
                if len(currentSample) > minFrame:
                    outSampleList.append(currentSample)
                    outFrameLabelList.append(currentLabel)
                    segLenList.append(len(currentSample))
    
            else:
                #remember invalid files
                pass
        return outSampleList, outFrameLabelList, segLenList
          

    def getActivityScore(self, skeletons):
        
        skeletons = self.filteringInvalidFrames(skeletons)
        frameNum, skeletonNum = skeletons.shape
        
        activitySocre = skeletons[1:, :] - skeletons[:-1, :]
                        
        return np.sum(np.abs(activitySocre))
    
    def extractRelativeJointFeatureFromDataList_spatial_temporal(self, sampleList, frameLabelList, usedJoints=np.linspace(0, 24, 25).astype('int')):

        outSampleList = list()
        outFrameLabelList = list()
        segLenList = list()
                
        jointNum = sampleList[0].shape[1]/3
        #selectedColumns = np.concatenate((usedJoints * 3, usedJoints * 3 + 1, usedJoints * 3 + 2))
        #selectedColumns.sort()
        #featureMat_all = np.zeros([1, ((jointNum * (jointNum)/2)*3) + ((jointNum ** 2) * 3)])
                
        for i in range(len(sampleList)):
            currentSample = sampleList[i]#[:, selectedColumns]
                    
            currentSpatialFeature = self.extractRelativeJointFeatureFromMat_spatial(currentSample, jointNum)
            currentTemporalFeature = self.extractRelativeJointFeatureFromMat_temporal(currentSample, jointNum)
                    
            currentFeature = np.concatenate((currentSpatialFeature[1: , :], currentTemporalFeature), axis = 1)
                
            outSampleList.append(currentFeature)
            outFrameLabelList.append(frameLabelList[i][1:])
            segLenList.append(currentFeature.shape[0])
                
                
        return outSampleList, outFrameLabelList, segLenList
    
    def getSample(self, data, minFrame, mConfig):


        if mConfig.skeletonSelection == 'longest one':
            outSample, outLabel, outSegLen = self.getData_longest(data, minFrame, mConfig)
        elif mConfig.skeletonSelection == 'longest two':
            outSample, outLabel, outSegLen = self.getData_longest_2(data, minFrame, mConfig)
            #mConfig.selectedJoint = np.concatenate([mConfig.selectedJoint,mConfig.selectedJoint])
        elif mConfig.skeletonSelection == "activest one":
            outSample, outLabel, outSegLen = self.getSample_activest(data, minFrame, mConfig)
        elif mConfig.skeletonSelection == 'activest two':
            outSample, outLabel, outSegLen = self.getSample_activest2(data, minFrame, mConfig)
            #mConfig.selectedJoint = np.concatenate([mConfig.selectedJoint,mConfig.selectedJoint])
        else:
            pass
        
        return outSample, outLabel, outSegLen
        
    def getSample_activest(self, data, minFrame, mConfig):
        usedJoints = mConfig.selectedJoint
        dimPerJoint = mConfig.dimPerJoint
        dataType = mConfig.dataType
        centralize = mConfig.centralize
        if centralize is True:
            centerJointInd = mConfig.centerJointInd
        selectedColumns = self.getDataColumnInd(usedJoints, dimPerJoint, dataType)
        selectedColumns.sort()
        outSampleList = list()
        outFrameLabelList = list()
        segLenList = list()
        
        jointNum = len(usedJoints)

        #sampleDim = len(selectedColumns)
        
        for i in range(len(data)):
            keys = data[i]['sample'].keys()
            if len(keys) != 0:
                
                #frameNum = data[i]['sample'][keys[0]].shape[0]
                
                activeScores = np.zeros(len(keys))
                # Find the longest one
                for keyIte in range(len(keys)):
                    currentSample = data[i]['sample'][keys[keyIte]][:,selectedColumns]
                    activeScores[keyIte] = self.getActivityScore(currentSample)
                    
                
                highestSampleKeyInd = activeScores.argmax()
                currentSample = data[i]['sample'][keys[highestSampleKeyInd]][:,selectedColumns]

                if centralize is True:
                    currentSample = currentSample - np.matlib.repmat(currentSample[:,centerJointInd * 3 : (centerJointInd + 1) * 3], 1, jointNum)
            

                currentLabel = np.zeros(len(currentSample))
                currentLabel[:] = int(data[i]['action']) - 1
                currentSample, currentLabel = self.filteringInvalidFrames(currentSample, currentLabel)
                
                if len(currentSample) > minFrame:
                    outSampleList.append(currentSample)
                    outFrameLabelList.append(currentLabel)
                    segLenList.append(len(currentSample))
    
            else:
                #remember invalid files
                pass
        return outSampleList, outFrameLabelList, segLenList
    
    
    
    
    def getData_vaeReconstruction(self, mConfig):
        
        def generateMask(segLenList, segLen_max):
            maskList = list()
            for i in range(len(segLenList)):
                maskList.append(np.zeros(segLen_max))
                maskList[i][1:segLenList[i]] = 1
            
            return maskList
        
        self.data_train, self.data_test, self.data_valid = self.splitData(mConfig)
        
        minimumLen = 1
        if mConfig.doHmm:
            minimumLen = 1 + mConfig.stateNumPerClass
            
            
        sampleList_train, frameLabelList_train, segLenList_train = self.getSample(self.data_train, minimumLen, mConfig)
        sampleList_test, frameLabelList_test, segLenList_test = self.getSample(self.data_test, minimumLen, mConfig)
        reconstructionSampleList_train, reconstructionFrameLabelList_train, reconstructionSegLenList_train = self.extractRelativeJointFeatureFromDataList_spatial_temporal(sampleList_train, frameLabelList_train, mConfig.selectedJoint)
        reconstructionSampleList_test, reconstructionFrameLabelList_test, reconstructionSegLenList_test = self.extractRelativeJointFeatureFromDataList_spatial_temporal(sampleList_test, frameLabelList_test, mConfig.selectedJoint)
        
        if mConfig.validation is True:
            sampleList_valid, frameLabelList_valid, segLenList_valid = self.getSample(self.data_valid, minimumLen, mConfig)
                
            reconstructionSampleList_valid, reconstructionFrameLabelList_valid, reconstructionSegLenList_valid = self.extractRelativeJointFeatureFromDataList_spatial_temporal(sampleList_valid, frameLabelList_valid, mConfig.selectedJoint)
        else:
            sampleList_valid = None 
            frameLabelList_valid = None 
            segLenList_valid = None    
            reconstructionSampleList_valid = None 
            reconstructionFrameLabelList_valid = None 
            reconstructionSegLenList_valid = None
    
    
        if mConfig.doNormalization is True:
            priorScalar = self.fitNormalizer(mConfig.normalizationStyle, \
                                             sampleList_train, segLenList_train, sampleList_test, segLenList_test, sampleList_valid, segLenList_valid)

        
        
            sampleList_train, sampleList_test, sampleList_valid = self.normalize(priorScalar, sampleList_train, sampleList_test)
            #reconstructionSampleList_train, reconstructionSampleList_test, reconstructionSampleList_valid = self.normalize(priorScalar1, reconstructionSampleList_train, reconstructionSampleList_test)
            
            
            mPrior = sio.loadmat('./prior.mat')
            mMean = mPrior['mean']
            mStd = mPrior['std']
            
            for i in range(len(sampleList_train)):
                sampleList_train[i] = sampleList_train[i] - mMean
                sampleList_train[i] = sampleList_train[i] / mStd
                
    
            for i in range(len(sampleList_test)):
                sampleList_test[i] = sampleList_test[i] - mMean
                sampleList_test[i] = sampleList_test[i] / mStd
                
    
                    
            if mConfig.validation is not None:
                for i in range(len(sampleList_test)):
                    sampleList_valid[i] = sampleList_valid[i] - mMean
                    sampleList_valid[i] = sampleList_valid[i] / mStd
                    



        segLenMat_train = np.array(segLenList_train)
        reconstructionSegLenMat_train = np.array(reconstructionSegLenList_train)
        
        segLenMat_test = np.array(segLenList_test)
        reconstructionSegLenMat_test = np.array(reconstructionSegLenList_test)
        
        if mConfig.validation:
            segLenMat_valid = np.array(segLenList_valid)
            reconstructionSegLenMat_valid = np.array(reconstructionSegLenList_valid)
            
            segLen_max = np.max([np.max(segLenMat_train), np.max(segLenMat_test), np.max(segLenMat_valid)])
            reconstructionSegLen_max = np.max([np.max(reconstructionSegLenMat_train), np.max(reconstructionSegLenMat_test), np.max(reconstructionSegLenMat_valid)])
        else:
            segLen_max = np.max([np.max(segLenMat_train), np.max(segLenMat_test)])
            reconstructionSegLen_max = np.max([np.max(reconstructionSegLenMat_train), np.max(reconstructionSegLenMat_test)])
    
        sampleList_train_padded, frameLabelList_train_padded, sampleList_test_padded, frameLabelList_test_padded, sampleList_valid_padded, frameLabelList_valid_padded = \
            self.padding(segLen_max, sampleList_train, frameLabelList_train, sampleList_test, frameLabelList_test, sampleList_valid, frameLabelList_valid)
            
            
        reconstructionSampleList_train_padded, reconstructionFrameLabelList_train_padded,  \
        reconstructionSampleList_test_padded, reconstructionFrameLabelList_test_padded, \
        reconstructionSampleList_valid_padded, reconstructionFrameLabelList_valid_padded = \
            self.padding(reconstructionSegLen_max, \
                         reconstructionSampleList_train, reconstructionFrameLabelList_train, \
                         reconstructionSampleList_test, reconstructionFrameLabelList_test, \
                         reconstructionSampleList_valid, reconstructionFrameLabelList_valid)


        
        maskList_train = generateMask(segLenList_train, segLen_max)
        maskList_test = generateMask(segLenList_test, segLen_max)
        if mConfig.validation:
            maskList_valid = generateMask(segLenList_valid, segLen_max)

        
        mConfig.reconstructionTargetDim = reconstructionSampleList_train[0].shape[-1]
        
        batchSource_train = list()
        batchSource_train.append(sampleList_train_padded)
        batchSource_train.append(frameLabelList_train_padded)
        batchSource_train.append(segLenList_train)
        
        batchSource_train.append(reconstructionSampleList_train_padded)
        batchSource_train.append(reconstructionFrameLabelList_train_padded)
        batchSource_train.append(reconstructionSegLenList_train)
        batchSource_train.append(maskList_train)
                
        batchSource_test = list()
        batchSource_test.append(sampleList_test_padded)
        batchSource_test.append(frameLabelList_test_padded)
        batchSource_test.append(segLenList_test)
        
        batchSource_test.append(reconstructionSampleList_test_padded)
        batchSource_test.append(reconstructionFrameLabelList_test_padded)
        batchSource_test.append(reconstructionSegLenList_test)
        batchSource_test.append(maskList_test)
            
        sampleDim = sampleList_train_padded[0].shape[-1]
        
        if mConfig.validation:

            batchSource_valid = list()
            batchSource_valid.append(sampleList_valid_padded)
            batchSource_valid.append(frameLabelList_valid_padded)
            batchSource_valid.append(segLenList_valid)
            
            batchSource_test.append(reconstructionSampleList_valid_padded)
            batchSource_test.append(reconstructionFrameLabelList_valid_padded)
            batchSource_test.append(reconstructionSegLenList_valid)
            batchSource_test.append(maskList_valid)
        
        else:
            batchSource_valid = None
    
        
        return batchSource_train, batchSource_test, batchSource_valid, segLen_max, sampleDim, \
            reconstructionFrameLabelList_train, reconstructionFrameLabelList_test, reconstructionFrameLabelList_valid
        
        
    
        
    def getData(self, mConfig):
        
        def generateMask(segLenList, segLen_max):
            maskList = list()
            for i in range(len(segLenList)):
                maskList.append(np.zeros(segLen_max))
                maskList[i][segLenList[i]-1] = 1
            
            return maskList
        
        self.data_train, self.data_test, self.data_valid = self.splitData(mConfig)
        
        if mConfig.dataset == 'G3DLie':
            mConfig.batchSize_train = len(self.data_train)
            mConfig.batchSize_test = len(self.data_train)
        
        minimumLen = 1
        if mConfig.doHmm:
            minimumLen = 1 + mConfig.stateNumPerClass
            
            
        sampleList_train, frameLabelList_train, segLenList_train = self.getSample(self.data_train, minimumLen, mConfig)
        sampleList_test, frameLabelList_test, segLenList_test = self.getSample(self.data_test, minimumLen, mConfig)
        if mConfig.validation is True:
            sampleList_valid, frameLabelList_valid, segLenList_valid = self.getSample(self.data_valid, minimumLen, mConfig)
        else:
            sampleList_valid = None 
            frameLabelList_valid = None 
            segLenList_valid = None  
        
        if mConfig.relativeJointFeature:
            sampleList_train, frameLabelList_train, segLenList_train = self.extractRelativeJointFeatureFromDataList_spatial_temporal(sampleList_train, frameLabelList_train, mConfig.selectedJoint)
            sampleList_test, frameLabelList_test, segLenList_test = self.extractRelativeJointFeatureFromDataList_spatial_temporal(sampleList_test, frameLabelList_test, mConfig.selectedJoint)
            
            if mConfig.validation:
                sampleList_valid, frameLabelList_valid, segLenList_valid = self.extractRelativeJointFeatureFromDataList_spatial_temporal(sampleList_valid, frameLabelList_valid, mConfig.selectedJoint)
                


        if mConfig.doNormalization is True:
            
            if mConfig.relativeJointFeature:
                                
                mPrior = sio.loadmat('./prior.mat')
                mMean = mPrior['mean']
                mStd = mPrior['std']
                
                for i in range(len(sampleList_train)):
                    sampleList_train[i] = sampleList_train[i] - mMean
                    sampleList_train[i] = sampleList_train[i] / mStd
                    
        
                for i in range(len(sampleList_test)):
                    sampleList_test[i] = sampleList_test[i] - mMean
                    sampleList_test[i] = sampleList_test[i] / mStd
                    
        
                        
                if mConfig.validation is not None:
                    for i in range(len(sampleList_test)):
                        sampleList_valid[i] = sampleList_valid[i] - mMean
                        sampleList_valid[i] = sampleList_valid[i] / mStd
                    
                
            else:
                
            
                priorScalar = self.fitNormalizer(mConfig.normalizationStyle, \
                                                 sampleList_train, segLenList_train, sampleList_test, segLenList_test, sampleList_valid, segLenList_valid)
       
            
                sampleList_train, sampleList_test, sampleList_valid = self.normalize(priorScalar, sampleList_train, sampleList_test)
            

        

        segLenMat_train = np.array(segLenList_train)
        segLenMat_test = np.array(segLenList_test)
        if mConfig.validation:
            segLenMat_valid = np.array(segLenList_valid)
            segLen_max = np.max([np.max(segLenMat_train), np.max(segLenMat_test), np.max(segLenMat_valid)])
            
        else:
            segLen_max = np.max([np.max(segLenMat_train), np.max(segLenMat_test)])
    
        sampleList_train_padded, frameLabelList_train_padded, sampleList_test_padded, frameLabelList_test_padded, sampleList_valid_padded, frameLabelList_valid_padded = \
            self.padding(segLen_max, sampleList_train, frameLabelList_train, sampleList_test, frameLabelList_test, sampleList_valid, frameLabelList_valid)
            

        
        batchSource_train = list()
        batchSource_train.append(sampleList_train_padded)
        batchSource_train.append(frameLabelList_train_padded)
        batchSource_train.append(segLenList_train)
                
        batchSource_test = list()
        batchSource_test.append(sampleList_test_padded)
        batchSource_test.append(frameLabelList_test_padded)
        batchSource_test.append(segLenList_test)
            
        sampleDim = sampleList_train_padded[0].shape[-1]
        
        if mConfig.validation:

            batchSource_valid = list()
            batchSource_valid.append(sampleList_valid_padded)
            batchSource_valid.append(frameLabelList_valid_padded)
            batchSource_valid.append(segLenList_valid)
        
        else:
            batchSource_valid = None
    
        
        return batchSource_train, batchSource_test, batchSource_valid, segLen_max, sampleDim, \
            frameLabelList_train, frameLabelList_test, frameLabelList_valid
        
        
        
        
        
        
        