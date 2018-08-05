'''
Created on Mar 21, 2018

@author: hshi
'''
from BatchLoader.BasicGeneralBatchLoader import BasicGeneralBatchLoader
import os
import numpy as np
class BaseNetwork(object):
    '''
    classdocs
    '''


    def __init__(self, sampleDim, mConfig):
        '''
        Constructor
        '''
        self.batchLoader_train = None
        self.batchLoader_valid = None
        self.batchLoader_test = None
        
        self.workingDir = self.mConfig.workingDir
        self.logFilePath = os.path.join(self.workingDir, 'log.txt')
        self.classNumAll = mConfig.classNum
        self.sampleDim = sampleDim

        
        self.maxGradient = self.mConfig.gradientClipThreshold
        self.NUM_EPOCHS_PER_DECAY = self.mConfig.NUM_EPOCHS_PER_DECAY
        self.LEARNING_RATE_DECAY_FACTOR = self.mConfig.LEARNING_RATE_DECAY_FACTOR
        self.baseLearningRate = self.mConfig.baseLearningRate
        self.finetune = self.mConfig.finetune
        
        
        
        
        #self.segLen_max = segLen_max
        #self.reconstructionTargetDim = sampleDim#self.mConfig.reconstructionTargetDim
        #self.hiddenDim_vae_z = sampleDim#self.mConfig.hiddenDim_vae_z
        #self.hiddenDim_vae_encoder = self.mConfig.hiddenDim_vae_encoder
        #self.hiddenDim_vae_decoder = self.mConfig.hiddenDim_vae_decoder
        
        
        #self.buildNet()
        
    def restore(self):
        if self.finetune is True:
            self.weightToSave = os.path.join(self.mConfig.workingDir, 'finetunedWeight.ckpt')

            # Finding finetune weight
            self.weightToRestore = os.path.join(self.mConfig.workingDir, 'finetunedWeight.ckpt.meta')
            if  os.path.exists(self.weightToRestore):
                self.weightToRestore = os.path.join(self.mConfig.workingDir, 'finetunedWeight.ckpt')

            else:
                self.weightToRestore = os.path.join(self.mConfig.workingDir, 'pretrainedWeight.ckpt')

            self.saver.restore(self.sess, self.weightToRestore)
            

        else:
            self.weightToSave = os.path.join(self.mConfig.workingDir, 'pretrainedWeight.ckpt')

            self.weightToRestore = os.path.join(self.mConfig.workingDir, 'pretrainedWeight.ckpt.meta')
            
            if os.path.exists(self.weightToRestore):
                self.weightToRestore = os.path.join(self.mConfig.workingDir, 'pretrainedWeight.ckpt')
                self.saver.restore(self.sess, self.weightToRestore)
                
        
    def buildNet(self):
        pass
    
    
        
        
    
    def updateLearningRate_expotenitialDecay(self, epochIte):
        self.currentLearningRate = self.baseLearningRate * (self.LEARNING_RATE_DECAY_FACTOR ** (epochIte / self.NUM_EPOCHS_PER_DECAY))
        return self.currentLearningRate
    
    
    def writeLog(self, str_):
        with open(self.logFilePath, "a") as f:
            f.write(str_)
            
            
    def initializingBatchLoaders(self,batchSize_train, data_train, 
                                batchSize_test, data_test,
                                batchSize_valid = None, data_valid = None):        
            
        self.batchLoader_train = BasicGeneralBatchLoader(batchSize_train, data_train)
        #self.batchNumPerEpoch_train = self.batchLoader_train.getBatchNumPerEpoch()
        #self.batchSize_train = batchSize_train
        
        
        self.batchLoader_test = None
        if batchSize_test is not None and data_test is not None:
            self.batchLoader_test = BasicGeneralBatchLoader(batchSize_test, data_test)
            #self.batchNumPerEpoch_test = self.batchLoader_test.getBatchNumPerEpoch()
            #self.batchSize_test = batchSize_test

    
        self.batchLoader_valid = None
        if batchSize_valid is None and data_valid is not None:
            self.batchLoader_valid = BasicGeneralBatchLoader(batchSize_valid, data_valid)
            #self.batchNumPerEpoch_valid = self.batchLoader_test.getBatchNumPerEpoch()
            #self.batchLoader_valid = batchSize_valid
            
    def run(self, sess, ops, feeds):
        
        return sess.run(ops, feed_dict = feeds)
    
    

        
    
    
    def train(self, maxTraingEpoch,
              batchSize_train, data_train, 
              batchSize_test, data_test,
              batchSize_valid = None, data_valid = None):
        
        if self.mConfig.crossSubject:
            self.writeLog(str(self.mConfig.subject_train))
            self.writeLog(str(self.mConfig.subject_test))
        
        self.initializingBatchLoaders(batchSize_train, data_train, batchSize_test, data_test, batchSize_valid, data_valid)
    
        overAllAccuracies_train = np.zeros(maxTraingEpoch)
        overAllLosses_train = np.zeros(maxTraingEpoch)
        overAllAccuracies_test = np.zeros(maxTraingEpoch)
        overAllLosses_test = np.zeros(maxTraingEpoch)    
        
        self.ops_train = [self.train_op, self.meanLoss, self.correctPredcitNum, self.orgGradientNorm]
        self.ops_vlaid = [self.meanLoss, self.correctPredcitNum]
        self.ops_test = [self.meanLoss, self.prediction, self.correctPredcitNum]
            
            
    
        for epochIte in range(maxTraingEpoch):
            
            self.updateLearningRate_expotenitialDecay(epochIte)
            #self.saver.save(self.sess, os.path.join(self.workingDir, 'model.ckpt'))
            if self.batchLoader_test is not None:
                tmpLoss_test, tmpAcc_test = self.testOneEpoch()
                
                overAllAccuracies_test[epochIte] = tmpAcc_test
                overAllLosses_test[epochIte] = tmpLoss_test
                
            if self.batchLoader_valid is not None:
                self.validOneEpoch()
            
            tmpLoss_train, tmpAcc_train = self.trainOneEpoch()
            overAllAccuracies_train[epochIte] = tmpAcc_train
            overAllLosses_train[epochIte] = tmpLoss_train
            
            
            # Do logging
            
            logStr = \
            'learning rate: ' + "{:.7f}".format(self.currentLearningRate) + '    ' +\
            'train_loss: ' + "{:.7f}".format(tmpLoss_train) + '    ' +\
            'test_loss:' + "{:.7f}".format(tmpLoss_test) + '    ' +\
            'train_accuracy: ' + "{:.7f}".format(tmpAcc_train) + '    ' +\
            'test_accuracy: ' + "{:.7f}".format(tmpAcc_test) + '\n'
            print (self.workingDir)
            print (logStr)
            self.writeLog(logStr)
            self.saver.save(self.sess, self.weightToSave)
            #self.saver
            
            
        return overAllAccuracies_train, overAllLosses_train, overAllAccuracies_test, overAllLosses_test
    
    
    def trainOneEpoch(self):
        pass
    
    def validOneEpoch(self):
        pass
        
    def testOneEpoch(self):
        pass
    
    
    
    
    