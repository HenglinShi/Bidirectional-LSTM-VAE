'''
Created on Apr 18, 2018

@author: hshi
'''
import os
from Experiment.ExperimengSettings import workingDir
from Configuration.BaseConfiguration import BaseConfiguration
dataset = 'chalearn'
import numpy as np
batchSize_train=100
batchSize_test=100

workingDir = os.path.join(workingDir, dataset)
if not os.path.exists(workingDir):
    os.mkdir(workingDir)

dataPath='/wrk/hshi/DONOTREMOVE/data/chaLearn2014'
classNum=20

dimPerJoint=9




blstmLayerNum=3
blstmLayerNeuronNum = [512,512,512] 
gradientClipThreshold=50
baseLearningRate=0.0005
NUM_EPOCHS_PER_DECAY=3
LEARNING_RATE_DECAY_FACTOR=0.999

selectedJoint=[0,1,2,3,4,5,6,7,8,9,10,11]
selectedJoint = np.array(selectedJoint)
doNormalization=True 
normalizationStyle='train'
centralize=False
#finetune=True
validation = False



def getConfig(experimentName, finetune):
    if experimentName.split('_')[0] == 'Blstm':
        hiddenDim_vae_encoder=256
        hiddenDim_vae_z=128
        hiddenDim_vae_decoder=256
        blstmLayerNeuronNum = [512,512,512] 
    
    
    elif experimentName.split('_')[1] == 'featureGeneration':
        hiddenDim_vae_encoder=128
        hiddenDim_vae_z=128
        hiddenDim_vae_decoder=256
        blstmLayerNeuronNum = [512,512,512] 
        
    
    elif experimentName.split('_')[2] == 'rawJoint':
        hiddenDim_vae_encoder=128
        hiddenDim_vae_z=128
        hiddenDim_vae_decoder=128
        
        if experimentName.split('_') == 'reconstructionClassification': 
            blstmLayerNeuronNum = [512,512,512] 
        else:
            blstmLayerNeuronNum = [512,512,512] 
        
    elif experimentName.split('_')[2] == 'relativeJoint':
        hiddenDim_vae_encoder=256
        hiddenDim_vae_z=128
        hiddenDim_vae_decoder=256

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
                                blstmLayerNum=blstmLayerNum,
                                blstmLayerNeuronNum=blstmLayerNeuronNum,
                                crossSubjectTest=False,
                                crossSubjectValidation=False,
                                subject_train=None,
                                subject_test=None,
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


#  if experimentName == 'MyVae_featureGeneration_reconstructionClassification':
#         hiddenDim_vae_encoder=128
#         hiddenDim_vae_z=128
#         hiddenDim_vae_decoder=256
#         
#         
#     elif experimentName == 'MyVae_featureGeneration_zClassification':
#         hiddenDim_vae_encoder=128
#         hiddenDim_vae_z=128
#         hiddenDim_vae_decoder=256
#         
#     elif experimentName == 'MyVae_selfReconstruction_rawJoint_reconstructionClassification':
#         hiddenDim_vae_encoder=128
#         hiddenDim_vae_z=128
#         hiddenDim_vae_decoder=128
#     elif experimentName == 'MyVae_selfReconstruction_rawJoint_zClassification':
#         hiddenDim_vae_encoder=128
#         hiddenDim_vae_z=128
#         hiddenDim_vae_decoder=128
#         
#     elif experimentName == 'MyVae_selfReconstruction_relativeJoint_reconstructionClassification':
#         hiddenDim_vae_encoder=256
#         hiddenDim_vae_z=128
#         hiddenDim_vae_decoder=256
#         
#     elif experimentName == 'MyVae_selfReconstruction_relativeJoint_zClassification':
#         hiddenDim_vae_encoder=256
#         hiddenDim_vae_z=128
#         hiddenDim_vae_decoder=256
#         
#     elif experimentName == 'StandardVae_featureGeneration_reconstructionClassification':
#         hiddenDim_vae_encoder=128
#         hiddenDim_vae_z=128
#         hiddenDim_vae_decoder=256
#         
#     elif experimentName == 'StandardVae_featureGeneration_zClassification':
#         hiddenDim_vae_encoder=128
#         hiddenDim_vae_z=128
#         hiddenDim_vae_decoder=256
#         
#         
#     elif experimentName == 'StandardVae_selfReconstruction_rawJoint_reconstructionClassification':
#         hiddenDim_vae_encoder=128
#         hiddenDim_vae_z=128
#         hiddenDim_vae_decoder=128
#         
#     elif experimentName == 'StandardVae_selfReconstruction_rawJoint_zClassification':
#         hiddenDim_vae_encoder=128
#         hiddenDim_vae_z=128
#         hiddenDim_vae_decoder=256
#         
#     elif experimentName == 'StandardVae_selfReconstruction_relativeJoint_reconstructionClassification':
#         hiddenDim_vae_encoder=256
#         hiddenDim_vae_z=128
#         hiddenDim_vae_decoder=256
#         
#     elif experimentName == 'StandardVae_selfReconstruction_relativeJoint_zClassification':
#         hiddenDim_vae_encoder=256
#         hiddenDim_vae_z=128
#         hiddenDim_vae_decoder=256