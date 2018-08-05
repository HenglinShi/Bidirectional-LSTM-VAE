'''
Created on Mar 19, 2018

@author: hshi
'''
from Dataset.Dataset import Dataset
import numpy as np
import os
from Tools.Tools import extractRelativeJointFeatureFromDataList_spatial_temporal

class UTKinect(Dataset):
    '''
    classdocs
    '''


    def __init__(self, dataDir, labelFilePath):
        '''
        Constructor
        '''
        Dataset.__init__(self)
        self.dataDir = dataDir
        self.labelFilePath = labelFilePath
        self.metaLabel, self.labelList = self.loadLabel()
        self.loadData()
        

    def loadData(self):
        
        self.dataList = list()
        data_all = list()
        segmentIter = 0

        for i in range(len(self.labelList)):
            dataName_currentVideo = self.labelList[i]['videoName']
            labels_currentVideo = self.labelList[i]['labels']
            
            sampleFilePath = os.path.join(self.dataDir, 'joints_' + dataName_currentVideo + '.txt')
            
            sample_currentVideo = np.loadtxt(sampleFilePath)
            
            data_currentVideo = list()
            
            for j in range(len(labels_currentVideo)):
                
                segmentIter += 1
                
                currentLabel = labels_currentVideo[j]
                begFrame = currentLabel['beginFrame']
                endFrame = currentLabel['endFrame']
                
                #sample_currentSegment = list()
                
                ind_1 = (sample_currentVideo[:,0] >= begFrame)
                ind_2 = (sample_currentVideo[:,0] <= endFrame)
                
                ind = ind_1 * ind_2
                
                ind = np.where(ind == True)[0]
                

                sampleMat_currentSegment = sample_currentVideo[ind, 1:]#np.array(sample_currentSegment)      
                currentFrameLabel = np.zeros([sampleMat_currentSegment.shape[0]]) 
                currentFrameLabel[:] = currentLabel['label']
                self.dataList.append({'videoName': dataName_currentVideo, 
                                      'sequenceID': segmentIter,
                                      'subjectId': currentLabel['subjectID'],
                                      'sample': sampleMat_currentSegment,
                                      'label': currentFrameLabel,
                                      'sequenceLabel': currentLabel['label']})
                
                
                data_currentVideo.append({'sequenceID': segmentIter,
                                 'sample': sampleMat_currentSegment,
                                 'label': currentLabel['label'],
                                 'subjectId': currentLabel['subjectID']})
             
            data_all.append(data_currentVideo)   
                
        return data_all
                        

        
    def loadFromFile(self, filePath):
        
        file_ = open(filePath)
        value_ = file_.read()
        file_.close()
        return value_
        
    def loadLabel(self):
        
        labelInfo_raw = self.loadFromFile(self.labelFilePath)
        
        labelInfo_raw = labelInfo_raw.split('\n')
        labelInfo_raw = filter(None, labelInfo_raw)
        
        
        labelTypes_all = list()
        labels_all = list()
        labels_currentVideo = list()
        dataName_currentVideo = None
        
        segmentNum = 0
        
        for i in range(len(labelInfo_raw)):
            #print i
            if ':' not in labelInfo_raw[i]:
                if dataName_currentVideo != None:
                    labels_all.append({'videoName': dataName_currentVideo,
                                       'labels': labels_currentVideo})
                    
                labels_currentVideo = list()
                dataName_currentVideo = labelInfo_raw[i]
            
            else:
                
                currentLabelInfo = labelInfo_raw[i]
                currentLabel, begFrame, endFrame = currentLabelInfo.split()[0:3]
                currentLabel = currentLabel[0:-1]
                currentSubject = dataName_currentVideo[1:3]
                
                if begFrame != 'NaN' and endFrame != 'NaN':
                    segmentNum += 1
                    print (dataName_currentVideo, begFrame, endFrame)
                    if currentLabel not in labelTypes_all:
                        labelTypes_all.append(currentLabel)
                        currentLabel = len(labelTypes_all)
                    else:
                        currentLabel = labelTypes_all.index(currentLabel) + 1
                    
                    labels_currentVideo.append({'label': currentLabel,
                                                'subjectID': currentSubject,
                                                'beginFrame': int(begFrame),
                                                'endFrame': int(endFrame)})
                
                
        labels_all.append({'videoName': dataName_currentVideo,
                           'labels': labels_currentVideo})
             
        metaData = np.zeros(shape = (segmentNum, 6), dtype = int)
    
        segmentIter = 0
        for i in range(len(labels_all)):
            dataName_currentVideo = labels_all[i]['videoName']
            labels_currentVideo = labels_all[i]['labels']
            
            for j in range(len(labels_currentVideo)):
                
                currentLabel = labels_currentVideo[j]
                #'subject': currentSuject
                metaData[segmentIter, :] = segmentIter + 1, \
                                           int(dataName_currentVideo[1:3]), \
                                           int(dataName_currentVideo[-2:]), \
                                           currentLabel['label'], \
                                           currentLabel['beginFrame'], \
                                           currentLabel['endFrame']
                
                segmentIter += 1
        
        # metadata segmentId//subjectID//iteratio//class/begfr/endfr
        return metaData, labels_all
    
