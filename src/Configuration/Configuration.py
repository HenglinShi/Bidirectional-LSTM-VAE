'''
Created on Mar 7, 2018

@author: hshi
'''

import ConfigParser
import numpy as np
class Configuration(object):
    '''
    classdocs
    '''


    def __init__(self, cfg_path):
        '''
        Constructor
        '''
        self.cfg_path = cfg_path
        cf = ConfigParser.ConfigParser()
        cf.read(self.cfg_path)
        
        self.skeletonNum = 1
        self.dataset = cf.get("config", "dataset")
        
        if self.dataset == 'chalearn' or self.dataset == 'chalearnLie':
            self.crossSubject = False
        
        elif self.dataset == 'ntu':
            self.dataRegen = cf.getboolean("config", 'dataRegen')
            self.crossSubject = cf.getboolean("config", "crossSubject")
            self.skeletonNum = cf.getint("config", "skeletonNum")
            self.skeletonSelection = cf.get("config", "skeletonSelection")
        
        elif self.dataset == 'msrAction3d' or self.dataset == 'G3DLie':
            self.crossSubject = cf.getboolean("config", "crossSubject")

        elif self.dataset == 'UTKinect':
            self.labelPath = cf.get("config", 'labelPath')
            self.crossSubject = cf.getboolean("config", "crossSubject")
        else:
            pass
        
        
        
        if self.crossSubject:
            self.subject_train = cf.get("config", "subject_train")
            self.subject_train = self.subject_train.split(' ')
            self.subject_test = cf.get("config", "subject_test")
            self.subject_test = self.subject_test.split(' ')
            
        self.dimPerJoint = cf.getint("config", "dimPerJoint")
        
        self.dataPath = cf.get("config", "dataPath")
        self.rawDataPath = cf.get("config", "rawDataPath")
        self.workingDir = cf.get("config", "workingDir")
        
        
        
        self.classNum = cf.getint("config", "classNum")
        self.selectedJoint = cf.get("config", "selectedJoint")
        self.selectedJoint = np.array(self.selectedJoint.split(' ')).astype('int')
        self.dataType = cf.get("config", "dataType")
        self.relativeJointFeature=cf.getboolean("config", "relativeJointFeature")   
        self.centralize=cf.get("config", "centralize")   
        if self.centralize is True:
            self.centerJointInd = cf.getint("config", "centerJointInd")
        
        self.doHmm = cf.getboolean("config", "doHmm")
        self.stateNumPerClass = cf.getint("config", "stateNumPerClass")
        

        self.batchSize_train = cf.getint("config", "batchSize_train")
        self.batchSize_test = cf.getint("config", "batchSize_test")
        self.validation = cf.getboolean("config", "validation")
        
        if self.validation:
            self.batchSize_valid =  cf.getint("config", "batchSize_train")
        
        self.doNormalization=cf.getboolean("config", "doNormalization")
        if self.doNormalization:
            self.normalizationStyle = cf.get("config", "normalizationStyle")
        
        self.netName = cf.get("config", "netName")
        self.mode = cf.get("config", "mode")
        self.gradientClipThreshold = cf.getfloat("config", "gradientClipThreshold")
        self.NUM_EPOCHS_PER_DECAY=cf.getint("config", "NUM_EPOCHS_PER_DECAY")
        self.LEARNING_RATE_DECAY_FACTOR=cf.getfloat("config", "LEARNING_RATE_DECAY_FACTOR")
        self.baseLearningRate = cf.getfloat("config", "baseLearningRate")
        self.finetune = cf.getboolean("config", "finetune")
        if self.finetune:
            self.weightToBeFinetuned=cf.get('config', 'trainedWeight')
        else:
            self.weightToBeFinetuned = None
            
            
        #if self.netName
        if self.netName.find('vae') >= 0:
            self.reconstructionTargetDim = cf.getint("config", "reconstructionTargetDim")
            self.hiddenDim_vae_encoder = cf.getint("config", "hiddenDim_vae_encoder")
            self.hiddenDim_vae_decoder = cf.getint("config", "hiddenDim_vae_decoder")
            self.hiddenDim_vae_z = cf.getint("config", "hiddenDim_vae_Z")
            pass
        
        if self.netName.lower().find('blstm') >= 0:
            self.blstmLayerNum=cf.getint("config", "blstmLayerNum")
            self.blstmLayerNeuronNum = cf.get("config", "blstmLayerNeuronNum")
            self.blstmLayerNeuronNum = np.array(self.blstmLayerNeuronNum.split(' ')).astype('int')
            