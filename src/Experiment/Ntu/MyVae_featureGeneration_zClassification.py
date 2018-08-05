'''
Created on Apr 18, 2018

@author: hshi
'''
import os
from Experiment.Ntu.ExperimentSettings import getConfig, workingDir
from Dataset.Ntu.Ntu import Ntu
from Model.VaeBlstm_v6 import VaeBlstm_v6
def main():
    args = parseArgs()
    experimentName = 'MyVae_featureGeneration_zClassification'
    finetune = args.finetune
    print "finetune " + finetune
    mConfig = getConfig(experimentName, finetune)
    dataset = 'chalearn'
    experimentDir = os.path.join(workingDir, experimentName)
    if not os.path.exists(experimentDir):
        os.mkdir(experimentDir)
        
    mConfig.workingDir = experimentDir
    
    mData = Ntu(mConfig.dataPath, mConfig)
    batchSource_train, batchSource_test, batchSource_valid, segLen_max, sampleDim, frameLabelList_train, frameLabelList_test, frameLabelList_valid = mData.getData_vaeReconstruction(mConfig)
    
    mNet = VaeBlstm_v6(sampleDim, segLen_max, mConfig)
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