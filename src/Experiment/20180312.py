'''
Created on Mar 7, 2018

@author: hshi
'''

import argparse
from Configuration.Configuration import Configuration
from DataWraper.NtuDataWraper import NtuDataWraper

from Model.VaeBlstm_v1 import VaeBlstm_v1
from Dataset.Chalearn.Chalearn import ChaLearn

def main():
    args = parseArgs()
    cfgPath = args.cfg_file
    
    mConfig = Configuration(cfgPath)
    network = mConfig.netName
    dataset = mConfig.dataset
    classNum = mConfig.classNum


    
    if dataset == 'ntu':
        
        batchSource_train, batchSource_test, segLen_max, sampleDim = NtuDataWraper(mConfig, 
                                                                                   mConfig.dataPath, 
                                                            mConfig.dataRegen, 
                                                            mConfig.crossSubject, 
                                                            mConfig.subject_train, 
                                                            mConfig.subject_test, 
                                                            mConfig.doHmm, 
                                                            mConfig.classNum, 
                                                            mConfig.stateNumPerClass, 
                                                            mConfig.relativeJointFeature, 
                                                            mConfig.doNormalization,
                                                            False)
        
            
    elif dataset == 'chalearn':
        chalearnData = ChaLearn(mConfig.dataPath, 1)
        batchSource_train, batchSource_test, batchSource_valid, segLen_max, sampleDim = chalearnData.getData_all(mConfig)
    
    elif dataset == 'utkinect':
        pass
    
    else: 
        pass
    
    
    
    if network == 'vaeBlstm_v1':
    
        mNet = VaeBlstm_v1(classNum,
                           segLen_max,
                           sampleDim = sampleDim,
                           baseLearningRate=mConfig.baseLearningRate,
                           maxGradient=mConfig.gradientClipThreshold,
                           workingDir=mConfig.workwingDir,
                           finetune = mConfig.finetune)    
         
        batchSize_train = mConfig.batchSize_train
        batchSize_test = mConfig.batchSize_test 
        
        mNet.train(600, 
                   batchSize_train, batchSource_train,
                   batchSize_test, batchSource_test)
    
    


def parseArgs():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', 
                        type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()