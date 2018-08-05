'''
Created on Apr 18, 2018

@author: hshi
'''
import os
from Configuration.Configuration import Configuration
from Dataset.Ntu.Ntu import Ntu
from Model.VaeBlstm_v5 import VaeBlstm_v5
def main():
    experimentName = 'MyVae_featureGeneration_reconstructionClassification'
    dataset = 'ntu'
    workingDir = ''
    dataPath = ''
    
    experimentDir = os.path.join(workingDir, dataset, experimentName)
    if not os.path.exists(os.path.join(workingDir, dataset)):
        os.mkdir(os.path.join(workingDir, dataset))
    os.mkdir(experimentDir)
    
    configPath = os.path.join(workingDir, 'config.cfg')
    mConfig = Configuration(configPath)
    
    mConfig.dataset = dataset
    mConfig.classNum = 60
    mConfig.dataPath = dataPath
    mConfig.dataRegen = False
    
    mData = Ntu(mConfig.dataPath, mConfig)
    batchSource_train, batchSource_test, batchSource_valid, segLen_max, sampleDim, frameLabelList_train, frameLabelList_test, frameLabelList_valid = mData.getData_vaeReconstruction(mConfig)
    
    mNet = VaeBlstm_v5(sampleDim, segLen_max, mConfig)
    mNet.train(6000, mConfig.batchSize_train, batchSource_train, mConfig.batchSize_test, batchSource_test)
    
    
    pass
if __name__ == '__main__':
    main()