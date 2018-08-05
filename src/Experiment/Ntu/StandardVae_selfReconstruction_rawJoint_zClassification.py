'''
Created on Apr 18, 2018

@author: hshi
'''
import os
from Configuration.Configuration import Configuration
from Dataset.Ntu.Ntu import Ntu
from Model.VaeBlstm_v11 import VaeBlstm_v11
from Experiment.Ntu.ExperimentSettings import getConfig, workingDir
def main():
    experimentName = 'StandardVae_selfReconstruction_rawJoint_zClassification'
    dataset = 'chalearn'
    #workingDir = '/wrk/hshi/DONOTREMOVE/git/FeatureLearningAndGestureRecognition/ExperimentArchive/New/20180418'
    experimentDir = os.path.join(workingDir, experimentName)
    if not os.path.exists(experimentDir):
        os.mkdir(experimentDir)
    args = parseArgs()
    finetune = args.finetune
    mConfig = getConfig(experimentName, finetune)
    
    mConfig.workingDir = experimentDir
    mConfig.baseLearningRate = 0.0001
    mData = Ntu(mConfig.dataPath, mConfig)
    batchSource_train, batchSource_test, batchSource_valid, segLen_max, sampleDim, frameLabelList_train, frameLabelList_test, frameLabelList_valid = mData.getData(mConfig)
    
    mNet = VaeBlstm_v11(sampleDim, segLen_max, mConfig)
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