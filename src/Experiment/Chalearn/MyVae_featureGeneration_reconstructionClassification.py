'''
Created on Apr 18, 2018

@author: hshi
'''
import argparse
import os
from Model.VaeBlstm_v5 import VaeBlstm_v5
from Experiment.Chalearn.ExperimentSettings import getConfig, workingDir
from Dataset.Chalearn.Chalearn import ChaLearn
def main():
    args = parseArgs()
    finetune = args.finetune
    print finetune
    experimentName = 'MyVae_featureGeneration_reconstructionClassification'
    if not os.path.exists(workingDir):
        os.mkdir(workingDir)
    
    experimentDir = os.path.join(workingDir, experimentName)
    if not os.path.exists(experimentDir):
        os.mkdir(experimentDir)
    
    #finetune = False
    mConfig = getConfig(experimentName, finetune)    
    mConfig.workingDir = experimentDir
    
    mData = ChaLearn(mConfig.dataPath, 1)

        
    batchSource_train, batchSource_test, batchSource_valid, segLen_max, sampleDim, frameLabelList_train, frameLabelList_test, frameLabelList_valid = mData.getData_vaeReconstruction(mConfig)
    
    mNet = VaeBlstm_v5(sampleDim, segLen_max, mConfig)
    mNet.train(6000, mConfig.batchSize_train, batchSource_train, mConfig.batchSize_test, batchSource_test)
    
    
    pass
def parseArgs():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    
    parser.add_argument('--finetune', dest='finetune',
                        help='optional config file', 
                        type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
    
