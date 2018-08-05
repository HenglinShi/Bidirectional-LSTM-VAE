import numpy as np
import random as rd
from abc import abstractmethod
from sklearn import preprocessing
import gc

class Dataset(object):
    '''
    classdocs
    '''
    def __init__(self):
        '''
        Constructor
        '''
        pass

    def filteringInvalidFrames(self, sample, frameLabel = None):
        validInds = np.where (np.abs(sample).sum(1) > 0)[0]
        if frameLabel is not None:  
            return sample[validInds, :], frameLabel[validInds]
        else: 
            return sample[validInds, :]
    
    def getSubjects(self, dataList):

        subjectList = list()
        for i in range(len(dataList)):
            
            currentSubject = dataList[i]['subjectId']
            if currentSubject not in subjectList:
                subjectList.append(currentSubject)
                
        return subjectList
    
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
    
    def getDataBySubjects(self, dataList, selectedSubjects):
        
        selectedData = list()
        
        for i in range(len(dataList)):
            currentSubject = dataList[i]['subjectId']
            
            if currentSubject in selectedSubjects:
                selectedData.append(dataList[i])
        return selectedData
    
    
    #@abstractmethod
    def splitData(self, mConfig):
        if mConfig.crossSubject is True:
            self.subjectList = self.getSubjects(self.dataList)
            self.subjectList_train, self.subjectList_test = self.crossSubjectSplitting(self.subjectList)
            
            #mConfig.subject_train = self.subjectList_train
            #mConfig.subject_test = self.subjectList_test
            
            self.data_train = self.getDataBySubjects(self.dataList, mConfig.subject_train)
            self.data_test = self.getDataBySubjects(self.dataList, mConfig.subject_test)
            
            if mConfig.validation is True:
                pass
            else:
                self.data_valid = None
                
    
        return self.data_train, self.data_test, self.data_valid
    
    
    
    def getDataColumnInd(self, usedJoints, dimPerJoint = 3, dataType = '3dCoordinate'):
        if dataType == '3dCoordinate rotation 2dCoordinate':
            selectedColumns = np.concatenate((usedJoints * dimPerJoint, usedJoints * dimPerJoint + 1, usedJoints * dimPerJoint + 2, \
                                              usedJoints * dimPerJoint + 3, usedJoints * dimPerJoint + 4, usedJoints * dimPerJoint + 5, \
                                              usedJoints * dimPerJoint + 6, usedJoints * dimPerJoint + 7, usedJoints * dimPerJoint + 8))
            
        elif dataType == '3dCoordinate':
            selectedColumns = np.concatenate((usedJoints * dimPerJoint, usedJoints * dimPerJoint + 1, usedJoints * dimPerJoint + 2))
            
        elif dataType == '3dCoordinate rotation':
            selectedColumns = np.concatenate((usedJoints * dimPerJoint, usedJoints * dimPerJoint + 1, usedJoints * dimPerJoint + 2, \
                                              usedJoints * dimPerJoint + 3, usedJoints * dimPerJoint + 4, usedJoints * dimPerJoint + 5, \
                                              usedJoints * dimPerJoint + 6))
            
        elif dataType == '3dCoordinate 2dCoordinate':
            selectedColumns = np.concatenate((usedJoints * dimPerJoint, usedJoints * dimPerJoint + 1, usedJoints * dimPerJoint + 2, \
                                              usedJoints * dimPerJoint + 7, usedJoints * dimPerJoint + 8))
        else:
            pass
        
        return selectedColumns
    
    def getSample(self, data, minFrame, mConfig):
        #usedJoints, dimPerJoint = 3, dataType = '3dCoordinate', centralize = False, centerJointInd = 1):
        
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
    
        for i in range(len(data)):
            currentSample = data[i]['sample'][:, selectedColumns]
            
            if centralize is True:
                currentSample = currentSample - np.matlib.repmat(currentSample[:,centerJointInd * 3 : (centerJointInd + 1) * 3], 1, jointNum)
            
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
    
    
    
    def frameLabelList2FrameStateList(self,frameLabelList, stateNumPerClass):
    
        frameStateList = list()
        
        for i in range(len(frameLabelList)):
            currentFrameLabels = frameLabelList[i]
            currentFrameStates = currentFrameLabels * stateNumPerClass + np.round(np.linspace(-0.5, stateNumPerClass - 0.51, currentFrameLabels.shape[0])).astype('int')
            frameStateList.append(currentFrameStates)
        
        return frameStateList
    
    
    def extractRelativeJointFeatureFromDataList_spatial_temporal(self, sampleList, frameLabelList, usedJoints=np.linspace(0, 24, 25).astype('int')):

        outSampleList = list()
        outFrameLabelList = list()
        segLenList = list()
                
        jointNum = len(usedJoints)
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
    
    def extractRelativeJointFeatureFromMat_spatial(self, sampleMat, jointNum):
        feature = np.zeros((sampleMat.shape[0], (jointNum * (jointNum - 1)/2)*3))
            
        featureColumnIte = 0
        for jointIte1 in range(jointNum - 1):
            for jointIte2 in range(jointIte1 + 1, jointNum):
                feature[:,featureColumnIte * 3 : (featureColumnIte + 1) * 3] = \
                    sampleMat[:, jointIte1 * 3 : (jointIte1 + 1) * 3] - sampleMat[:, jointIte2 * 3 : (jointIte2 + 1) * 3]
                featureColumnIte += 1
            
        return feature
        
        
        
    def extractRelativeJointFeatureFromMat_temporal(self, sampleMat, jointNum):
            
        feature = np.zeros((sampleMat.shape[0] - 1, (jointNum ** 2) * 3))
            
        featureColumnIte = 0
            
        for jointIte1 in range(jointNum):
            for jointIte2 in range(jointNum):
                feature[:, featureColumnIte * 3 : (featureColumnIte + 1) * 3] = \
                    sampleMat[1:, jointIte1 * 3 : (jointIte1 + 1) * 3] - sampleMat[0:-1, jointIte2 * 3 : (jointIte2 + 1) * 3]
                    
                featureColumnIte += 1
                    
        return feature


    def fitNormalizer(self, normalizationStyle, \
                  sampleList_train, segLenList_train, \
                  sampleList_test, segLenList_test, \
                  sampleList_valid = None, segLenList_valid = None):
            
        if normalizationStyle == 'train':
            sampleMat = self.list2Mat_fast(sampleList_train, segLenList_train)
            priorScalar = preprocessing.StandardScaler().fit(sampleMat)
            del sampleMat
                
        elif normalizationStyle == 'train test':

                
            sampleMat_train = self.list2Mat_fast(sampleList_train, segLenList_train)
            
            sampleMat_test = self.list2Mat_fast(sampleList_test, segLenList_test)
            sampleMat = np.concatenate((sampleMat_train, sampleMat_test))
                
            priorScalar = preprocessing.StandardScaler().fit(sampleMat)
            del sampleMat, sampleMat_train, sampleMat_test
                
        elif normalizationStyle == 'train valid':
                
            sampleMat_train = self.list2Mat_fast(sampleList_train, segLenList_train)
            sampleMat_valid = self.list2Mat_fast(sampleList_valid, segLenList_valid)
            sampleMat = np.concatenate((sampleMat_train, sampleMat_valid))
                
            priorScalar = preprocessing.StandardScaler().fit(sampleMat)
            del sampleMat, sampleMat_train, sampleMat_valid
                
        elif normalizationStyle == 'train test valid':
            
            sampleMat_train = self.list2Mat_fast(sampleList_train, segLenList_train)
            sampleMat_valid = self.list2Mat_fast(sampleList_valid, segLenList_valid)
            sampleMat_test = self.list2Mat_fast(sampleList_test, segLenList_test)
            sampleMat = np.concatenate((sampleMat_train, sampleMat_valid, sampleMat_test))
            priorScalar = preprocessing.StandardScaler().fit(sampleMat)
            del sampleMat, sampleMat_train, sampleMat_valid, sampleMat_test
                
        gc.collect()    
            
        return priorScalar
    
    def normalize(self, priorScalar, sampleList_train, sampleList_test, sampleList_valid = None):
        sampleList_train = self.basicNormalization(sampleList_train, priorScalar)
        
        sampleList_test = self.basicNormalization(sampleList_test, priorScalar)
                
        if sampleList_valid is not None:
            sampleList_valid = self.basicNormalization(sampleList_train, priorScalar)
                    
        return sampleList_train, sampleList_test, sampleList_valid
    
    def basicNormalization(self, sampleList, prior):
    
        for i in range(len(sampleList)):
            sampleList[i] = prior.transform(sampleList[i])
    
        return sampleList
    
    def list2Mat_fast(self, inList, sequenceLengthsList):
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

    def padding(self, segLen_max, \
            sampleList_train, frameLabelList_train, \
            sampleList_test, frameLabelList_test,\
            sampleList_valid= None, frameLabelList_valid = None):
    
    
        sampleList_train_padded = self.paddingData(sampleList_train, segLen_max)
        frameLabelList_train_padded = self.paddingData(frameLabelList_train, segLen_max)
            
        sampleList_test_padded = self.paddingData(sampleList_test, segLen_max)
        frameLabelList_test_padded = self.paddingData(frameLabelList_test, segLen_max)
            
        if sampleList_valid is not None:
            sampleList_valid_padded = self.paddingData(sampleList_valid, segLen_max)
            frameLabelList_valid_padded = self.paddingData(frameLabelList_valid, segLen_max)
            
        else:
            sampleList_valid_padded = None
            frameLabelList_valid_padded = None
                
        return sampleList_train_padded, frameLabelList_train_padded, sampleList_test_padded, frameLabelList_test_padded, sampleList_valid_padded, frameLabelList_valid_padded
    
    
    def paddingData(self, sampleList, segLen_max):
        
        if sampleList is None:
            return None
        
        paddedDataList = list()
        sampleNum = len(sampleList)
        dataShape = list(sampleList[0].shape)
        dataShape[0] = segLen_max
        
        for sampleIte in range(sampleNum):
            currentPaddedData = np.zeros(dataShape)
            currentPaddedData[:sampleList[sampleIte].shape[0],...] = sampleList[sampleIte]
            paddedDataList.append(currentPaddedData)
            
        return paddedDataList
    
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
   
            priorScalar1 = self.fitNormalizer(mConfig.normalizationStyle, \
                                              reconstructionSampleList_train, reconstructionSegLenList_train, \
                                              reconstructionSampleList_test, reconstructionSegLenList_test, \
                                              reconstructionSampleList_valid, reconstructionSegLenList_valid)
        
        
            sampleList_train, sampleList_test, sampleList_valid = self.normalize(priorScalar, sampleList_train, sampleList_test)
            reconstructionSampleList_train, reconstructionSampleList_test, reconstructionSampleList_valid = self.normalize(priorScalar1, reconstructionSampleList_train, reconstructionSampleList_test)
            
            
        else:
            pass
            
        if mConfig.doHmm is True:
            mConfig.classNum = mConfig.classNum * mConfig.stateNumPerClass
            frameLabelList_train = self.frameLabelList2FrameStateList(frameLabelList_train, mConfig.stateNumPerClass)
            frameLabelList_test = self.frameLabelList2FrameStateList(frameLabelList_test, mConfig.stateNumPerClass)
            reconstructionFrameLabelList_train = self.frameLabelList2FrameStateList(reconstructionFrameLabelList_train, mConfig.stateNumPerClass)
            reconstructionFrameLabelList_test = self.frameLabelList2FrameStateList(reconstructionFrameLabelList_test, mConfig.stateNumPerClass)
            
            if mConfig.validation:
                frameLabelList_valid = self.frameLabelList2FrameStateList(frameLabelList_valid, mConfig.stateNumPerClass)
                reconstructionFrameLabelList_valid = self.frameLabelList2FrameStateList(reconstructionFrameLabelList_valid, mConfig.stateNumPerClass)
            else:
                frameLabelList_valid = None
                reconstructionFrameLabelList_valid = None 
            
        else:
            pass
        
        

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
            priorScalar = self.fitNormalizer(mConfig.normalizationStyle, \
                                             sampleList_train, segLenList_train, sampleList_test, segLenList_test, sampleList_valid, segLenList_valid)
   

        
        
            sampleList_train, sampleList_test, sampleList_valid = self.normalize(priorScalar, sampleList_train, sampleList_test)
            
            
        else:
            pass
            
        if mConfig.doHmm is True:
            mConfig.classNum = mConfig.classNum * mConfig.stateNumPerClass
            frameLabelList_train = self.frameLabelList2FrameStateList(frameLabelList_train, mConfig.stateNumPerClass)
            frameLabelList_test = self.frameLabelList2FrameStateList(frameLabelList_test, mConfig.stateNumPerClass)
            
            if mConfig.validation:
                frameLabelList_valid = self.frameLabelList2FrameStateList(frameLabelList_valid, mConfig.stateNumPerClass)
            else:
                frameLabelList_valid = None
            
        else:
            pass
        
        

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
        
        

