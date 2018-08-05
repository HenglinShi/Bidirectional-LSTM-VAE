'''
Created on Apr 18, 2018

@author: hshi
'''
import os
from Configuration.Configuration import Configuration
from Dataset.Chalearn.Chalearn import ChaLearn
from Experiment.Chalearn.ExperimentSettings import getConfig, workingDir
from Model.VaeBlstm_v16 import VaeBlstm_v16
def main():
    experimentName = 'Fc_selfReconstruction_rawJoint_zClassification'
    dataset = 'chalearn'
    #workingDir = '/wrk/hshi/DONOTREMOVE/git/FeatureLearningAndGestureRecognition/ExperimentArchive/New/20180418'
    experimentDir = os.path.join(workingDir, experimentName)
    if not os.path.exists(experimentDir):
        os.mkdir(experimentDir)
    args = parseArgs()
    finetune = args.finetune
    print (finetune)
    mConfig = getConfig(experimentName, finetune)
    
    mConfig.workingDir = experimentDir
    mConfig.baseLearningRate = 0.0002
    mData = ChaLearn(mConfig.dataPath, 1)

        
    batchSource_train, batchSource_test, batchSource_valid, segLen_max, sampleDim, frameLabelList_train, frameLabelList_test, frameLabelList_valid = mData.getData(mConfig)
    
    mNet = VaeBlstm_v16(sampleDim, segLen_max, mConfig)
    mNet.train(6000, mConfig.batchSize_train, batchSource_train, mConfig.batchSize_test, batchSource_test)

import argparse

def parseArgs():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    
    parser.add_argument('--finetune', dest='finetune',
                        help='optional config file', 
                        type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()