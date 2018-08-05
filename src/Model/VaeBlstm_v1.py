'''
Created on Feb 21, 2018

@author: hshi
'''
import tensorflow as tf
from Model.Model1 import myModel3, vaeLoss_forLstm,\
    blstmSequencePredictionNetwork
from Model.Model1 import oneLayerBidirectionalLstmDecoder, oneLayerBidirectionalLstmEncoder, vaeSampler
from BatchLoader.BasicGeneralBatchLoader import BasicGeneralBatchLoader
import numpy as np
from numpy.random import RandomState
import os
seed = 42
np_rng = RandomState(seed)
class VaeBlstm_v1(object):
    '''
    classdocs
    '''


    def __init__(self,
                 stateNumAll,
                 segLen_max,
                 sampleDim,
                 reconstructionTargetDim=75,
                 hiddenDim_vae_encoder=200,
                 hiddenDim_vae_decoder=200,
                 hiddenDim_vae_z=75,
                 baseLearningRate = 0.0005,
                 maxGradient = 15,
                 NUM_EPOCHS_PER_DECAY = 8,
                 LEARNING_RATE_DECAY_FACTOR = 0.99,
                 workingDir = None,
                 finetune = False,
                 mConfig = None):
        '''
        Constructor
        '''
        
        
        self.batchLoader_train = None
        self.batchLoader_valid = None
        self.batchLoader_test = None
        self.batchNumPerEpoch_train = None
        self.batchNumPerEpoch_valid = None
        self.batchNumPerEpoch_test = None
        
        self.batchSize_train = None
        self.batchSize_valid = None
        self.batchSize_test = None
        self.mConfig = mConfig
        if mConfig is not None:
            self.workingDir = self.mConfig.workingDir
            self.reconstructionTargetDim = sampleDim#self.mConfig.reconstructionTargetDim
            self.hiddenDim_vae_z = sampleDim#self.mConfig.hiddenDim_vae_z
            self.hiddenDim_vae_encoder = self.mConfig.hiddenDim_vae_encoder
            self.hiddenDim_vae_decoder = self.mConfig.hiddenDim_vae_decoder
            self.maxGradient = self.mConfig.gradientClipThreshold
            self.NUM_EPOCHS_PER_DECAY = self.mConfig.NUM_EPOCHS_PER_DECAY
            self.LEARNING_RATE_DECAY_FACTOR = self.mConfig.LEARNING_RATE_DECAY_FACTOR
            self.baseLearningRate = self.mConfig.baseLearningRate
            self.finetune = self.mConfig.finetune
        
        
        else:
            self.workingDir = workingDir
            self.reconstructionTargetDim = sampleDim
            self.hiddenDim_vae_z = sampleDim
            
            self.hiddenDim_vae_encoder = hiddenDim_vae_encoder
            self.hiddenDim_vae_decoder = hiddenDim_vae_decoder
            self.maxGradient = maxGradient
            self.NUM_EPOCHS_PER_DECAY = NUM_EPOCHS_PER_DECAY
            self.LEARNING_RATE_DECAY_FACTOR = LEARNING_RATE_DECAY_FACTOR
            self.baseLearningRate = baseLearningRate
            self.finetune = finetune
            
        
        
        self.stateNumAll = stateNumAll
        self.segLen_max = segLen_max
        self.sampleDim = sampleDim
        
        self.logFilePath = os.path.join(self.workingDir, 'log.txt')
        
        self.buildNet()

    def updateLearningRate_expotenitialDecay(self, epochIte):
        self.currentLearningRate = self.baseLearningRate * (self.LEARNING_RATE_DECAY_FACTOR ** (epochIte / self.NUM_EPOCHS_PER_DECAY))
        return self.currentLearningRate
    
    def writeLog(self, str_):
        with open(self.logFilePath, "a") as f:
            f.write(str_)
            
    
    def buildInputs(self):
        
        self.placeHolder_samples = tf.placeholder("float", [None, self.segLen_max, self.sampleDim])
        self.placeHolder_targets_reconstruction = tf.placeholder("float", [None, self.segLen_max, self.reconstructionTargetDim])
        self.placeHolder_noises = tf.placeholder("float", [None, self.segLen_max, self.hiddenDim_vae_z])
        
        self.placeHolder_labels = tf.placeholder(tf.int64, [None])
        
        self.placeHolder_seqLens = tf.placeholder(tf.int64, [None])
        self.placeHoder_learningRate = tf.placeholder("float", [])
        
        self.vectorizedLabels = tf.one_hot(self.placeHolder_labels, self.stateNumAll)
        
        #self.vectorizedLabels_flatten = tf.reshape(self.vectorizedLabels, [-1, self.stateNumAll])
        self.mask = tf.reshape(tf.sequence_mask(self.placeHolder_seqLens, self.segLen_max, dtype = tf.float32), [-1])
        
        
    def buildVae(self): 
        
        self.vae_mean, self.vae_SE_ln = oneLayerBidirectionalLstmEncoder(self.placeHolder_samples, \
                                                                         self.placeHolder_seqLens, \
                                                                         self.hiddenDim_vae_encoder, \
                                                                         self.hiddenDim_vae_z)
        
        self.Z = vaeSampler(self.vae_mean, self.vae_SE_ln, self.placeHolder_noises)
        self.reconstruction_vae = oneLayerBidirectionalLstmDecoder(self.Z, self.placeHolder_seqLens, self.hiddenDim_vae_decoder, self.reconstructionTargetDim)
    
        
        # Action NEtwork
        # reshaping reconstruction back.
        #reconstruction_vae_withTN = tf.reshape(reconstruction_vae, shape = [-1, segLen_max, actionFeatureDim])
        
        
    def buildBlstm(self):
        self.prediction = blstmSequencePredictionNetwork(self.Z, self.placeHolder_seqLens, self.segLen_max, self.stateNumAll)
    
        
        #EVALUATION
        
        self.correctPrediction = tf.equal(tf.argmax(self.prediction,1), tf.argmax(self.vectorizedLabels, 1))
        #self.maskedCorrectPrediction = tf.multiply(self.mask, tf.cast(self.correctPrediction, tf.float32))
        self.correctPredcitNum = tf.reduce_sum(tf.cast(self.correctPrediction, tf.int64))
        
        
    
    
    def buildLoss(self):
        self.vae_mean_squared = tf.square(self.vae_mean, name = "vae_encoder_mean_squared")
        self.vae_SE = tf.exp(self.vae_SE_ln, name = "vae_encoder_SE")
        self.loss_vae, _, _ = vaeLoss_forLstm(self.reconstruction_vae, self.placeHolder_targets_reconstruction, self.vae_mean_squared, self.vae_SE, self.vae_SE_ln)
        self.maskedVaeLoss = tf.multiply(self.mask, self.loss_vae)
        self.meanVaeLoss = tf.divide(tf.reduce_sum(self.maskedVaeLoss), tf.reduce_sum(self.mask))
        
        
        self.loss_stateClassification = tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.vectorizedLabels)
        self.meanClassificationLoss = tf.reduce_mean(self.loss_stateClassification)
        
        self.meanLoss = self.meanVaeLoss + self.meanClassificationLoss

        
        
        #self.maskedLoss = tf.multiply(self.mask, self.loss_total)
        
        
        #self.meanLoss = tf.divide(tf.reduce_sum(self.maskedLoss), tf.reduce_sum(self.mask))
        
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.placeHoder_learningRate)
        
        self.params = tf.trainable_variables()
        self.gradients = tf.gradients(self.meanLoss, self.params)
        self.clippedGradients, self.orgGradientNorm = tf.clip_by_global_norm(self.gradients, self.maxGradient)
        self.train_op = self.optimizer.apply_gradients(zip(self.clippedGradients, self.params))
        self.init = tf.global_variables_initializer()
        
        

    def buildNet(self):
        
        self.buildInputs()
        self.buildVae()
        self.buildBlstm()
        self.buildLoss()
        self.saver = tf.train.Saver(max_to_keep=None)
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))
        self.sess.run(self.init)
        if self.finetune:
            self.saver.restore(self.sess, os.path.join(self.workingDir, 'model.ckpt'))
    
    def initializingBatchLoaders(self,batchSize_train, data_train, 
                                batchSize_test, data_test,
                                batchSize_valid = None, data_valid = None):        
            
        self.batchLoader_train = BasicGeneralBatchLoader(batchSize_train, data_train)
        self.batchNumPerEpoch_train = self.batchLoader_train.getBatchNumPerEpoch()
        self.batchSize_train = batchSize_train
        
        
        self.batchLoader_test = None
        if batchSize_test is not None and data_test is not None:
            self.batchLoader_test = BasicGeneralBatchLoader(batchSize_test, data_test)
            self.batchNumPerEpoch_test = self.batchLoader_test.getBatchNumPerEpoch()
            self.batchSize_test = batchSize_test

    
        self.batchLoader_valid = None
        if batchSize_valid is None and data_valid is not None:
            self.batchLoader_valid = BasicGeneralBatchLoader(batchSize_valid, data_valid)
            self.batchNumPerEpoch_valid = self.batchLoader_test.getBatchNumPerEpoch()
            self.batchLoader_valid = batchSize_valid
        
        
            
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
        self.ops_test = [self.meanLoss, self.correctPredcitNum]
            
            
    
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
            self.saver.save(self.sess, os.path.join(self.workingDir, 'model.ckpt'))
            #self.saver
            
            
        return overAllAccuracies_train, overAllLosses_train, overAllAccuracies_test, overAllLosses_test
    
    
    
    def run(self, sess, ops, feeds):
        
        return sess.run(ops, feed_dict = feeds)
    
    
    def trainOneEpoch(self):
        correctPredictNum_currentTrainEpoch = np.zeros(self.batchNumPerEpoch_train)
        sequenceLengths_currentTrainEpoch = np.zeros(self.batchNumPerEpoch_train)
        loss_currentTrainEpoch = 0.0
                
        for batchIte in range(self.batchNumPerEpoch_train):
            
            [sampleBatch_train, frameLabelBatch_train, seqlenBatch_train] = self.batchLoader_train.getNextBatch()
                    
            sampleBatch_train = np.array(sampleBatch_train)
            frameLabelBatch_train = np.array(frameLabelBatch_train)
            seqlenBatch_train = np.array(seqlenBatch_train)
            noise = np_rng.normal(0, 0.05, (int(self.batchSize_train), int(self.segLen_max), int(self.hiddenDim_vae_z))).astype(np.float)
                   
                    
            currentFeed = {self.placeHolder_samples: sampleBatch_train, self.placeHolder_targets_reconstruction: sampleBatch_train, \
                           self.placeHolder_noises: noise, self.placeHolder_labels: frameLabelBatch_train[:,0], \
                           self.placeHolder_seqLens: seqlenBatch_train, self.placeHoder_learningRate: self.currentLearningRate}
                    
                    
            _, currentLoss_train, currentCorrectPredcitNum_train, weightNorm = self.run(self.sess, self.ops_train, currentFeed)       

                    
                    
            loss_currentTrainEpoch += currentLoss_train
            correctPredictNum_currentTrainEpoch[batchIte] = currentCorrectPredcitNum_train

            sequenceLengths_currentTrainEpoch[batchIte] = seqlenBatch_train.shape[0]#.sum()
                     
        return [loss_currentTrainEpoch, \
                correctPredictNum_currentTrainEpoch.sum() * 1.0 / sequenceLengths_currentTrainEpoch.sum()]
    
    
    def test(self):
        
        
        pass
        
        
    def testOneEpoch(self):
        
        correctPredictNum_currentTestEpoch = np.zeros(self.batchNumPerEpoch_test)
        sequenceLengths_currentTestEpoch = np.zeros(self.batchNumPerEpoch_test)
        loss_currentTestEpoch = 0.0

                

        for batchIte in range(self.batchNumPerEpoch_test):
            [sampleBatch_test, frameLabelBatch_test, seqlenBatch_test] = self.batchLoader_test.getNextBatch()
                     
            noise = np.zeros((self.batchSize_test, self.segLen_max, self.hiddenDim_vae_z)).astype(np.float)
                     
            sampleBatch_test = np.array(sampleBatch_test)
            frameLabelBatch_test = np.array(frameLabelBatch_test)
            
            
            
            seqlenBatch_test = np.array(seqlenBatch_test)
                     
                     
            currentFeed = {self.placeHolder_samples: sampleBatch_test, self.placeHolder_targets_reconstruction: sampleBatch_test, \
                           self.placeHolder_noises: noise, self.placeHolder_labels: frameLabelBatch_test[:,0], \
                           self.placeHolder_seqLens: seqlenBatch_test}
            
            currentLoss_test, currentCorrectPredcitNum_test = self.run(self.sess, self.ops_test, currentFeed)
     
            loss_currentTestEpoch += currentLoss_test
            correctPredictNum_currentTestEpoch[batchIte] = currentCorrectPredcitNum_test
            sequenceLengths_currentTestEpoch[batchIte] = seqlenBatch_test.shape[0]#.sum()
                     
        
        return [loss_currentTestEpoch, correctPredictNum_currentTestEpoch.sum() * 1.0 / sequenceLengths_currentTestEpoch.sum()]


        
        
    def validOneEpoch(self):
        pass
        
        
        
        

        