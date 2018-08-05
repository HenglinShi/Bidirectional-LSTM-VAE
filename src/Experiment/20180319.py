'''
Created on Mar 7, 2018

@author: hshi
'''

import argparse
from Configuration.Configuration import Configuration
#from DataWraper.NtuDataWraper import NtuDataWraper
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from Model.VaeBlstm_v1 import VaeBlstm_v1
from Dataset.Chalearn.Chalearn import ChaLearn
#from Dataset.MsrAction3d.MsrAction3d import MsrAction3d
#from Dataset.UTKinect.UTKinect import UTKinect
from Model.VaeBlstm_v2 import VaeBlstm_v2
from Model.VaeBlstm_v3 import VaeBlstm_v3
from Model.VaeBlstm_v4 import VaeBlstm_v4
from Tools.HmmDecoder import HmmDecoder
from Dataset.Ntu.Ntu import Ntu
from Model.Blstm_v1 import Blstm_v1
from Model.VaeBlstm_v5 import VaeBlstm_v5
from Model.VaeBlstm_v6 import VaeBlstm_v6

from Model.VaeBlstm_v7 import VaeBlstm_v7
from Model.VaeBlstm_v8 import VaeBlstm_v8
from Dataset.MsrAction3d.MsrAction3d import MsrAction3d
from Dataset.UTKinect.UTKinect import UTKinect
from Dataset.Chalearn.ChalearnLie import ChalearnLie
from Model.VaeBlstm_v9 import VaeBlstm_v9
from Dataset.G3D.G3DLie import G3DLie
from Model.VaeBlstm_v11 import VaeBlstm_v11
from Model.VaeBlstm_v12 import VaeBlstm_v12
def main():
    args = parseArgs()
    cfgPath = args.cfg_file
    
    mConfig = Configuration(cfgPath)
    network = mConfig.netName
    dataset = mConfig.dataset
    classNum = mConfig.classNum
    
    if dataset == 'G3DLie':
        md = G3DLie(mConfig)
        if network == 'Blstm_v1':
            batchSource_train, batchSource_test, _, segLen_max, sampleDim, _, _, _  = md.getData(mConfig)

    elif dataset == 'chalearnLie':
        md = ChalearnLie(mConfig)
        if network == 'Blstm_v1':
            batchSource_train, batchSource_test, segLen_max, sampleDim, \
            reconstructionFrameLabelList_train, reconstructionFrameLabelList_test = md.getData(mConfig)
        else:
            batchSource_train, batchSource_test, segLen_max, sampleDim, \
            reconstructionFrameLabelList_train, reconstructionFrameLabelList_test = md.getData_vaeReconstruction_lie(mConfig)
    
    elif dataset == 'ntu':

        ntuData = Ntu(mConfig.dataPath, mConfig)
        
        if network == 'vaeBlstm_v4' or network == 'vaeBlstm_v5' or network == 'vaeBlstm_v6':
            batchSource_train, batchSource_test, batchSource_valid, segLen_max, sampleDim, frameLabelList_train, frameLabelList_test, frameLabelList_valid = ntuData.getData_vaeReconstruction(mConfig)
        else:
            batchSource_train, batchSource_test, batchSource_valid, segLen_max, sampleDim, frameLabelList_train, frameLabelList_test, frameLabelList_valid =ntuData.getData(mConfig)
    elif dataset == 'chalearn':
        
        chalearnData = ChaLearn(mConfig.dataPath, 1)
        if network == 'vaeBlstm_v4' or network == 'vaeBlstm_v5' or network == 'vaeBlstm_v6':
            batchSource_train, batchSource_test, batchSource_valid, segLen_max, sampleDim, frameLabelList_train, frameLabelList_test, frameLabelList_valid = chalearnData.getData_vaeReconstruction(mConfig)
            
        else:
            batchSource_train, batchSource_test, batchSource_valid, segLen_max, sampleDim, frameLabelList_train, frameLabelList_test, frameLabelList_valid = chalearnData.getData(mConfig)
            
            
    elif dataset == 'UTKinect':
        utkinectData = UTKinect(mConfig.dataPath, mConfig.labelPath)
        
        
        if network == 'vaeBlstm_v4' or network == 'vaeBlstm_v5' or network == 'vaeBlstm_v6':
            
            batchSource_train, batchSource_test, batchSource_valid, segLen_max, sampleDim, frameLabelList_train, frameLabelList_test, frameLabelList_valid = utkinectData.getData_vaeReconstruction(mConfig)
            
        else: 
            batchSource_train, batchSource_test, batchSource_valid, segLen_max, sampleDim, frameLabelList_train, frameLabelList_test, frameLabelList_valid = utkinectData.getData(mConfig)
    
    
        mConfig.batchSize_train = len(frameLabelList_train)
        mConfig.batchSize_test = len(frameLabelList_test)
    
    elif dataset == 'msrAction3d':
        msrAction3dData = MsrAction3d(mConfig.rawDataPath)
        
        if network == 'vaeBlstm_v4' or network == 'vaeBlstm_v5' or network == 'vaeBlstm_v6':
            batchSource_train, batchSource_test, batchSource_valid, segLen_max, sampleDim, frameLabelList_train, frameLabelList_test, frameLabelList_valid = msrAction3dData.getData_vaeReconstruction(mConfig)
            
        else:
            batchSource_train, batchSource_test, batchSource_valid, segLen_max, sampleDim, frameLabelList_train, frameLabelList_test, frameLabelList_valid = msrAction3dData.getData(mConfig)
    
        mConfig.batchSize_train = len(frameLabelList_train)
        mConfig.batchSize_test = len(frameLabelList_test)
    
    else: 
        pass
    
    
    
    if network == 'vaeBlstm_v1':
    
        mNet = VaeBlstm_v1(classNum, segLen_max, sampleDim = sampleDim, baseLearningRate=mConfig.baseLearningRate, maxGradient=mConfig.gradientClipThreshold, workingDir=mConfig.workingDir, finetune = mConfig.finetune, mConfig=mConfig)    
        mNet.train(6000, mConfig.batchSize_train, mConfig.batchSource_train, mConfig.batchSize_test, batchSource_test)
    
    
    elif network == 'vaeBlstm_v2':
        mNet = VaeBlstm_v2(sampleDim, segLen_max, mConfig)
        mNet.train(6000, mConfig.batchSize_train, batchSource_train, mConfig.batchSize_test, batchSource_test)
        
    elif network == 'vaeBlstm_v3':
        mNet = VaeBlstm_v3(sampleDim, segLen_max, mConfig)
        mNet.train(6000, mConfig.batchSize_train, batchSource_train, mConfig.batchSize_test, batchSource_test)


    elif network == 'vaeBlstm_v4':
        mNet = VaeBlstm_v4(sampleDim, segLen_max, mConfig)
        
        if mConfig.mode == 'Train':
            mNet.train(6000, mConfig.batchSize_train, batchSource_train, mConfig.batchSize_test, batchSource_test)
            
        else:
            batchSize_test = mConfig.batchSize_test 
            prediction, groundtruth = mNet.getPrediction_currentTestEpoch(batchSize_test, batchSource_test)
            decoder = HmmDecoder(frameLabelList_train, mConfig.classNum)
            correctPrediction_ = 0
            
            for i in range(len(batchSource_test[0])):
                [currentPrediction, _, _] = decoder.decode(prediction[i])
                correctPrediction_ += int(currentPrediction[0]/mConfig.stateNumPerClass == int(groundtruth[i])/mConfig.stateNumPerClass)
            print correctPrediction_
            print correctPrediction_*1.0/len(batchSource_test[0])
        # Test part
        # Test part the batch size will be set to 1
        
        #mNet.batchLoader_test.reset()
        
    elif network == 'Blstm_v1':
        mNet = Blstm_v1(sampleDim, segLen_max, mConfig)
        mNet.train(6000, mConfig.batchSize_train, batchSource_train, mConfig.batchSize_test, batchSource_test)

    elif network == 'vaeBlstm_v5':
        mNet = VaeBlstm_v5(sampleDim, segLen_max, mConfig)
        mNet.train(6000, mConfig.batchSize_train, batchSource_train, mConfig.batchSize_test, batchSource_test)
        
    elif network == 'vaeBlstm_v6':
        mNet = VaeBlstm_v6(sampleDim, segLen_max, mConfig)
        mNet.train(6000, mConfig.batchSize_train, batchSource_train, mConfig.batchSize_test, batchSource_test)
        
    elif network == 'vaeBlstm_v7':
        mNet = VaeBlstm_v7(sampleDim, segLen_max, mConfig)
        mNet.train(6000, mConfig.batchSize_train, batchSource_train, mConfig.batchSize_test, batchSource_test)
        
    elif network == 'vaeBlstm_v8':
        mNet = VaeBlstm_v8(sampleDim, segLen_max, mConfig)
        mNet.train(6000, mConfig.batchSize_train, batchSource_train, mConfig.batchSize_test, batchSource_test)
        
    elif network == 'vaeBlstm_v9':
        mNet = VaeBlstm_v9(sampleDim, segLen_max, mConfig)
        mNet.train(6000, mConfig.batchSize_train, batchSource_train, mConfig.batchSize_test, batchSource_test)
        
    elif network == 'vaeBlstm_v11':
        mNet = VaeBlstm_v11(sampleDim, segLen_max, mConfig)
        mNet.train(6000, mConfig.batchSize_train, batchSource_train, mConfig.batchSize_test, batchSource_test)

    elif network == 'vaeBlstm_v12':
        mNet = VaeBlstm_v12(sampleDim, segLen_max, mConfig)
        mNet.train(6000, mConfig.batchSize_train, batchSource_train, mConfig.batchSize_test, batchSource_test)

def parseArgs():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', 
                        type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()