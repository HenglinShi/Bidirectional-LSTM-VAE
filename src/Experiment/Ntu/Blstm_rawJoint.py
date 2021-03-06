'''
Created on Apr 18, 2018

@author: hshi
'''
import os
from Model.Blstm_v1 import Blstm_v1
from Experiment.Ntu.ExperimentSettings import getConfig, workingDir
from Dataset.Ntu.Ntu import Ntu
def main():
    experimentName = 'Blstm_rawJoint'
    mConfig = getConfig(experimentName, 'False')
    
    dataset = 'chalearn'
    #workingDir = '/wrk/hshi/DONOTREMOVE/git/FeatureLearningAndGestureRecognition/ExperimentArchive/New/20180418'
    experimentDir = os.path.join(workingDir, experimentName)
    if not os.path.exists(experimentDir):
        os.mkdir(experimentDir)
    
    mConfig.workingDir = experimentDir

        
    mConfig.baseLearningRate = 0.0002
    mData = Ntu(mConfig.dataPath, mConfig)
    batchSource_train, batchSource_test, batchSource_valid, segLen_max, sampleDim, frameLabelList_train, frameLabelList_test, frameLabelList_valid = mData.getData(mConfig)
    
    mNet = Blstm_v1(sampleDim, segLen_max, mConfig)
    mNet.train(6000, mConfig.batchSize_train, batchSource_train, mConfig.batchSize_test, batchSource_test)


if __name__ == '__main__':
    main()