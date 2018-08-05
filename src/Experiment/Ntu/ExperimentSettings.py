'''
Created on Apr 19, 2018

@author: hshi
'''
import os
from Experiment.ExperimengSettings import workingDir
from Configuration.BaseConfiguration import BaseConfiguration
dataset = 'ntu'
workingDir = os.path.join(workingDir, dataset)
if not os.path.exists(workingDir):
    os.mkdir(workingDir)

import numpy as np

batchSize_train=100
batchSize_test=100


selectedJoint= [0,1,2,3,4,5,6,7,8,9,10,11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
selectedJoint = np.array(selectedJoint)
crossSubject=True

subject_train='001,002,004,005,008,009,013,014,015,016,017,018,019,025,027,028,031,034,035,038'
subject_train = subject_train.split(',')
subject_test='003,006,007,010,011,012,020,021,022,023,024,026,029,030,032,033,036,037,039,040'
subject_test = subject_test.split(',')
classNum=60
dataPath='/wrk/hshi/DONOTREMOVE/workspaces/caffe-workspace/BMVC_submission/Data/ntu.pkl'
#dataPath='/wrk/hshi/DONOTREMOVE/workspaces/caffe-workspace/NtuDataset/ntu_part_1_1_1_1_1_1_1_1_1.pkl'

dimPerJoint=12
dataRegen=False

skeletonSelection='activest one'
gradientClipThreshold=50
baseLearningRate=0.0005
NUM_EPOCHS_PER_DECAY=8
LEARNING_RATE_DECAY_FACTOR=0.99


doNormalization=True 
normalizationStyle='train test'
centerJointInd=1
centralize=False

validation=False
batchSize_train=100
batchSize_test=100





gradientClipThreshold=50
baseLearningRate=0.0005
NUM_EPOCHS_PER_DECAY=3
LEARNING_RATE_DECAY_FACTOR=0.999

doNormalization=True 
centralize=False
finetune=True
validation = False

def getConfig(experimentName, finetune):
    
    if experimentName.split('_')[0] == 'Blstm':
        hiddenDim_vae_encoder=256
        hiddenDim_vae_z=128
        hiddenDim_vae_decoder=256
        blstmLayerNeuronNum = [512,512,512] 
    
    
    elif experimentName.split('_')[1] == 'featureGeneration':
        hiddenDim_vae_encoder=256
        hiddenDim_vae_z=256
        hiddenDim_vae_decoder=512
        blstmLayerNeuronNum = [512,512,512] 
        
    
    elif experimentName.split('_')[2] == 'rawJoint':
        hiddenDim_vae_encoder=256
        hiddenDim_vae_z=256
        hiddenDim_vae_decoder=256
        
        if experimentName.split('_') == 'reconstructionClassification': 
            blstmLayerNeuronNum = [512,512,512] 
        else:
            blstmLayerNeuronNum = [512,512,512] 
        
    elif experimentName.split('_')[2] == 'relativeJoint':
        hiddenDim_vae_encoder=512
        hiddenDim_vae_z=256
        hiddenDim_vae_decoder=512

        if experimentName.split('_') == 'reconstructionClassification': 
            blstmLayerNeuronNum = [512,512,512] 
        else:
            blstmLayerNeuronNum = [512,512,512] 
    

    if finetune == 'False':
        finetune = False
        print ('dasdsadsadsadsadsadsadsadsadsadsadsadsadsadsadasd')
        
    else:
        finetune = True
    
    mConfig = BaseConfiguration(workingDir=None,
                                dataset='chalearn',
                                dataPath=dataPath,
                                classNum=classNum,
                                selectedJoint=selectedJoint,
                                dimPerJoint=dimPerJoint,
                                relativeJointFeature=False,
                                finetune=finetune,
                                doNormalization=doNormalization,
                                normalizationStyle=normalizationStyle,
                                batchSize_train=batchSize_train,
                                batchSize_test=batchSize_test,
                                validation=validation,
                                batchSize_valid=batchSize_test,
                                hiddenDim_vae_encoder=hiddenDim_vae_encoder,
                                hiddenDim_vae_decoder=hiddenDim_vae_decoder,
                                hiddenDim_vae_z=hiddenDim_vae_z,
                                blstmLayerNum=len(blstmLayerNeuronNum),
                                blstmLayerNeuronNum=blstmLayerNeuronNum,
                                crossSubjectTest=crossSubject,
                                crossSubjectValidation=False,
                                subject_train=subject_train,
                                subject_test=subject_test,
                                subject_valid=None,
                                gradientClipThreshold=gradientClipThreshold,
                                baseLearningRate=baseLearningRate,
                                NUM_EPOCHS_PER_DECAY=NUM_EPOCHS_PER_DECAY,
                                LEARNING_RATE_DECAY_FACTOR=LEARNING_RATE_DECAY_FACTOR)
    
    if mConfig.finetune is True:
        mConfig.baseLearningRate = 0.0002
        mConfig.NUM_EPOCHS_PER_DECAY= 2
        mConfig.LEARNING_RATE_DECAY_FACTOR= 0.999
        
    else:
        mConfig.baseLearningRate = 0.0005
        mConfig.NUM_EPOCHS_PER_DECAY= 2
        mConfig.LEARNING_RATE_DECAY_FACTOR= 0.999
        
        
    
    return mConfig
