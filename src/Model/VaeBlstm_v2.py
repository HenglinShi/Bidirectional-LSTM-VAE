'''
Created on Mar 21, 2018

@author: hshi
'''
import os
from Model.BaseNetwork import BaseNetwork
import tensorflow as tf
import numpy as np
from Model.Model1 import oneLayerBidirectionalLstmEncoder, vaeSampler,\
    oneLayerBidirectionalLstmDecoder, blstmSequencePredictionNetwork,\
    vaeLoss_forLstm, blstmFramePredictionNetwork
    
from numpy.random import RandomState
from Configuration.Configuration import Configuration
seed = 42
np_rng = RandomState(seed)

class VaeBlstm_v2(BaseNetwork):
    '''
    classdocs
    '''


    def __init__(self, sampleDim, segLen_max, mConfig):
        '''
        Constructor
        '''
        self.mConfig = mConfig
        super(VaeBlstm_v2, self).__init__(sampleDim, mConfig)
        self.segLen_max = segLen_max
        self.reconstructionTargetDim = sampleDim#self.mConfig.reconstructionTargetDim
        self.hiddenDim_vae_z = sampleDim#self.mConfig.hiddenDim_vae_z
        self.hiddenDim_vae_encoder = self.mConfig.hiddenDim_vae_encoder
        self.hiddenDim_vae_decoder = self.mConfig.hiddenDim_vae_decoder
        
        
        self.buildNet()
        
    def buildNet(self):
        #BaseNetwork.buildNet(self)
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

        
        
        
    def buildInputs(self):
        
        self.placeHolder_samples = tf.placeholder("float", [None, self.segLen_max, self.sampleDim])
        self.placeHolder_targets_reconstruction = tf.placeholder("float", [None, self.segLen_max, self.reconstructionTargetDim])
        self.placeHolder_noises = tf.placeholder("float", [None, self.segLen_max, self.hiddenDim_vae_z])
        
        self.placeHolder_seqLens = tf.placeholder(tf.int64, [None])
        self.placeHoder_learningRate = tf.placeholder("float", [])
        self.mask = tf.reshape(tf.sequence_mask(self.placeHolder_seqLens, self.segLen_max, dtype = tf.float32), [-1])
        
        self.placeHolder_labels = tf.placeholder(tf.int64, [None, self.segLen_max])
        #self.vectorizedLabels = tf.one_hot(self.placeHolder_labels, self.classNumAll)
        
        self.vectorizedLabels = tf.reshape(tf.one_hot(self.placeHolder_labels, self.classNumAll), 
                                           [-1, self.classNumAll])
        
        
            
        
        
        
    def buildVae(self):
        self.vae_mean, self.vae_SE_ln = oneLayerBidirectionalLstmEncoder(self.placeHolder_samples, \
                                                                         self.placeHolder_seqLens, \
                                                                         self.hiddenDim_vae_encoder, \
                                                                         self.hiddenDim_vae_z)
        
        self.Z = vaeSampler(self.vae_mean, self.vae_SE_ln, self.placeHolder_noises)
        self.reconstruction_vae = oneLayerBidirectionalLstmDecoder(self.Z, self.placeHolder_seqLens, self.hiddenDim_vae_decoder, self.reconstructionTargetDim)
    
        
    def buildBlstm(self):
        self.prediction = blstmFramePredictionNetwork(self.Z, self.placeHolder_seqLens, self.segLen_max, self.classNumAll, self.mConfig.blstmLayerNum, self.mConfig.blstmLayerNeuronNum)
    
        self.correctPrediction = tf.equal(tf.argmax(self.prediction,1), tf.argmax(self.vectorizedLabels, 1))
        self.maskedCorrectPrediction = tf.multiply(self.mask, tf.cast(self.correctPrediction, tf.float32))
        self.correctPredcitNum = tf.reduce_sum(tf.cast(self.maskedCorrectPrediction, tf.int64))
        

        
        
    def buildLoss(self):
        
        self.vae_mean_squared = tf.square(self.vae_mean, name = "vae_encoder_mean_squared")
        self.vae_SE = tf.exp(self.vae_SE_ln, name = "vae_encoder_SE")
        
        self.loss_vae, _, _ = vaeLoss_forLstm(self.reconstruction_vae, self.placeHolder_targets_reconstruction, self.vae_mean_squared, self.vae_SE, self.vae_SE_ln)
        self.loss_stateClassification = tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.vectorizedLabels)
        #self.loss_stateClassification = tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=vectorizedLabels_flatten)
        self.loss_overall = self.loss_vae + self.loss_stateClassification
        self.loss_maskedOverall = tf.multiply(self.mask, self.loss_overall)
        
        
        self.meanLoss = tf.divide(tf.reduce_sum(self.loss_maskedOverall), tf.reduce_sum(self.mask))
        
        
        #self.meanClassificationLoss = tf.reduce_mean(self.loss_stateClassification)
        #self.meanLoss = self.meanVaeLoss + self.meanClassificationLoss
        #self.maskedLoss = tf.multiply(self.mask, self.loss_total)
        #self.meanLoss = tf.divide(tf.reduce_sum(self.maskedLoss), tf.reduce_sum(self.mask))
        
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.placeHoder_learningRate)
        
        self.params = tf.trainable_variables()
        self.gradients = tf.gradients(self.meanLoss, self.params)
        self.clippedGradients, self.orgGradientNorm = tf.clip_by_global_norm(self.gradients, self.maxGradient)
        self.train_op = self.optimizer.apply_gradients(zip(self.clippedGradients, self.params))
        self.init = tf.global_variables_initializer()
        
        
        
        
    def trainOneEpoch(self):
        correctPredictNum_currentTrainEpoch = np.zeros(self.batchLoader_train.getBatchNumPerEpoch())
        sequenceLengths_currentTrainEpoch = np.zeros(self.batchLoader_train.getBatchNumPerEpoch())
        loss_currentTrainEpoch = 0.0
                
        for batchIte in range(self.batchLoader_train.getBatchNumPerEpoch()):
            
            [sampleBatch_train, frameLabelBatch_train, seqlenBatch_train] = self.batchLoader_train.getNextBatch()
                    
            sampleBatch_train = np.array(sampleBatch_train)
            frameLabelBatch_train = np.array(frameLabelBatch_train)
            seqlenBatch_train = np.array(seqlenBatch_train)
            noise = np_rng.normal(0, 0.05, (int(self.batchLoader_train.batchSize), int(self.segLen_max), int(self.hiddenDim_vae_z))).astype(np.float)
                   
                    
            currentFeed = {self.placeHolder_samples: sampleBatch_train, self.placeHolder_targets_reconstruction: sampleBatch_train, \
                           self.placeHolder_noises: noise, self.placeHolder_labels: frameLabelBatch_train, \
                           self.placeHolder_seqLens: seqlenBatch_train, self.placeHoder_learningRate: self.currentLearningRate}
                    
                    
            _, currentLoss_train, currentCorrectPredcitNum_train, weightNorm = self.run(self.sess, self.ops_train, currentFeed)       

                    
                    
            loss_currentTrainEpoch += currentLoss_train
            correctPredictNum_currentTrainEpoch[batchIte] = currentCorrectPredcitNum_train

            sequenceLengths_currentTrainEpoch[batchIte] = seqlenBatch_train.sum()#shape[0]#.sum()
                     
        return [loss_currentTrainEpoch, \
                correctPredictNum_currentTrainEpoch.sum() * 1.0 / sequenceLengths_currentTrainEpoch.sum()]
    
    
    
    def testOneEpoch(self):
        correctPredictNum_currentTestEpoch = np.zeros(self.batchLoader_test.getBatchNumPerEpoch())
        sequenceLengths_currentTestEpoch = np.zeros(self.batchLoader_test.getBatchNumPerEpoch())
        loss_currentTestEpoch = 0.0

                

        for batchIte in range(self.batchLoader_test.getBatchNumPerEpoch()):
            [sampleBatch_test, frameLabelBatch_test, seqlenBatch_test] = self.batchLoader_test.getNextBatch()
                     
            noise = np.zeros((self.batchLoader_test.batchSize, self.segLen_max, self.hiddenDim_vae_z)).astype(np.float)
                     
            sampleBatch_test = np.array(sampleBatch_test)
            frameLabelBatch_test = np.array(frameLabelBatch_test)
            
            
            
            seqlenBatch_test = np.array(seqlenBatch_test)
                     
                     
            currentFeed = {self.placeHolder_samples: sampleBatch_test, self.placeHolder_targets_reconstruction: sampleBatch_test, \
                           self.placeHolder_noises: noise, self.placeHolder_labels: frameLabelBatch_test, \
                           self.placeHolder_seqLens: seqlenBatch_test}
            
            currentLoss_test, currentCorrectPredcitNum_test = self.run(self.sess, self.ops_test, currentFeed)
     
            loss_currentTestEpoch += currentLoss_test
            correctPredictNum_currentTestEpoch[batchIte] = currentCorrectPredcitNum_test
            sequenceLengths_currentTestEpoch[batchIte] = seqlenBatch_test.sum()#.shape[0]#.sum()
                     
        
        return [loss_currentTestEpoch, correctPredictNum_currentTestEpoch.sum() * 1.0 / sequenceLengths_currentTestEpoch.sum()]



        
        
        
        
        
        
        
        
        
        
        
        
        
        