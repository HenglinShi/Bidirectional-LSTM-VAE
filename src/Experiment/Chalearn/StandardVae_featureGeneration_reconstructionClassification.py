'''
Created on Apr 18, 2018

@author: hshi
'''
import os
from Experiment.Chalearn.ExperimentSettings import getConfig, workingDir
from Dataset.Chalearn.Chalearn import ChaLearn
from Model.VaeBlstm_v13 import VaeBlstm_v13
def main():
    experimentName = 'StandardVae_featureGeneration_reconstructionClassification'

    args = parseArgs()
    finetune = args.finetune
    mConfig = getConfig(experimentName, finetune)
    
    dataset = 'chalearn'
    
    experimentDir = os.path.join(workingDir, experimentName)
    if not os.path.exists(experimentDir):
        os.mkdir(experimentDir)
    
    mConfig.workingDir = experimentDir

        
        
        
    mData = ChaLearn(mConfig.dataPath, 1)

    batchSource_train, batchSource_test, batchSource_valid, segLen_max, sampleDim, frameLabelList_train, frameLabelList_test, frameLabelList_valid = mData.getData_vaeReconstruction(mConfig)
    
    mNet = VaeBlstm_v13(sampleDim, segLen_max, mConfig)
    mNet.train(6000, mConfig.batchSize_train, batchSource_train, mConfig.batchSize_test, batchSource_test)


import argparse
def parseArgs():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    
    parser.add_argument('--finetune', dest='finetune',
                        help='optional config file', 
                        type=str)

    args = parser.parse_args()
    return args    
    
    pass
if __name__ == '__main__':
    main()