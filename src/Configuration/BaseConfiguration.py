'''
Created on Apr 18, 2018

@author: hshi
'''

class BaseConfiguration(object):
    '''
    classdocs
    '''


    def __init__(self,
                 workingDir = None,
                 dataset = None,
                 dataPath = None,
                 classNum = None,
                 selectedJoint = None,
                 dimPerJoint = None,
                 relativeJointFeature = None,
                 finetune = None,
                 doNormalization = None,
                 normalizationStyle = None,
                 batchSize_train = None,
                 batchSize_test = None,
                 validation = None,
                 batchSize_valid = None,
                 hiddenDim_vae_encoder = None,
                 hiddenDim_vae_decoder = None,
                 hiddenDim_vae_z = None,
                 blstmLayerNum = None,
                 blstmLayerNeuronNum = None,
                 crossSubjectTest = None,
                 crossSubjectValidation = None,
                 subject_train = None,
                 subject_test = None,
                 subject_valid = None,
                 gradientClipThreshold = None,
                 baseLearningRate = None,
                 NUM_EPOCHS_PER_DECAY = None,
                 LEARNING_RATE_DECAY_FACTOR = None):
        '''
        Constructor
        '''
        self.workingDir = workingDir
        self.dataset = dataset
        self.dataPath = dataPath
        self.classNum = classNum
        self.selectedJoint = selectedJoint
        self.dimPerJoint = dimPerJoint
        
        self.validation = validation
        self.batchSize_train = batchSize_train
        self.batchSize_test = batchSize_test
        self.batchSize_valid = batchSize_valid
        
        self.crossSubjectTest = crossSubjectTest
        self.crossSubjectValidation = crossSubjectValidation
        self.subject_train = subject_train
        self.subject_test = subject_test
        self.subject_valid = subject_valid
        
        self.relativeJointFeature = relativeJointFeature
        
        self.doNormalization = doNormalization
        self.normalizationStyle = normalizationStyle
        
        self.finetune = finetune
        
        self.hiddenDim_vae_encoder = hiddenDim_vae_encoder
        self.hiddenDim_vae_decoder = hiddenDim_vae_encoder
        self.hiddenDim_vae_z = hiddenDim_vae_z
        self.blstmLayerNum = blstmLayerNum
        self.blstmLayerNeuronNum = blstmLayerNeuronNum
        
        
        self.gradientClipThreshold = gradientClipThreshold
        self.baseLearningRate = baseLearningRate
        self.NUM_EPOCHS_PER_DECAY = NUM_EPOCHS_PER_DECAY
        self.LEARNING_RATE_DECAY_FACTOR = LEARNING_RATE_DECAY_FACTOR

        self.doHmm = False
        self.dataType = '3dCoordinate'
        self.centralize = False
        self.crossSubject = self.crossSubjectTest
        self.dataRegen = False
        
        self.skeletonSelection = 'activest one'
        
        