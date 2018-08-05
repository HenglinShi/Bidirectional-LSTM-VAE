'''
Created on Apr 4, 2018

@author: hshi
'''
from BaseNetwork import BaseNetwork
import tensorflow as tf
from Model.Model1 import blstmFramePredictionNetwork,\
    blstmSequencePredictionNetwork
import numpy as np
from BatchLoader.BasicGeneralBatchLoader import BasicGeneralBatchLoader
class Blstm_v1(BaseNetwork):
    '''
    classdocs
    '''


    def __init__(self, sampleDim, segLen_max, mConfig):
        '''
        Constructor
        '''
        self.mConfig = mConfig
        super(Blstm_v1, self).__init__(sampleDim, mConfig)
        self.segLen_max = segLen_max
        self.buildNet()
        
        
    def buildNet(self):
        self.buildInputs()
        self.buildBlstm()
        self.buildLoss()
        self.saver = tf.train.Saver(max_to_keep=None)
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))
        self.sess.run(self.init)
        self.restore()
      
            
    def buildInputs(self):
        
        self.placeHolder_samples = tf.placeholder("float", [None, self.segLen_max, self.sampleDim])
        self.placeHolder_seqLens = tf.placeholder(tf.int64, [None])
        self.placeHoder_learningRate = tf.placeholder("float", [])

        self.placeHolder_labels = tf.placeholder(tf.int64, [None])
        self.vectorizedLabels = tf.one_hot(self.placeHolder_labels, self.classNumAll)
        
        
    def buildBlstm(self):
        self.prediction = blstmSequencePredictionNetwork(self.placeHolder_samples, self.placeHolder_seqLens, self.segLen_max, \
                                                      self.classNumAll, self.mConfig.blstmLayerNum, self.mConfig.blstmLayerNeuronNum)
    
        self.correctPrediction = tf.equal(tf.argmax(self.prediction,1), tf.argmax(self.vectorizedLabels, 1))
        self.correctPredcitNum = tf.reduce_sum(tf.cast(self.correctPrediction, tf.int64))
    
    def buildLoss(self):
        
        self.loss_classification = tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.vectorizedLabels)
        self.meanLoss = tf.reduce_mean(self.loss_classification)

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
                   
                    
            currentFeed = {self.placeHolder_samples: sampleBatch_train, \
                           self.placeHolder_labels: frameLabelBatch_train[:,0], \
                           self.placeHolder_seqLens: seqlenBatch_train, \
                           self.placeHoder_learningRate: self.currentLearningRate}
                    
                    
            _, currentLoss_train, currentCorrectPredcitNum_train, weightNorm = self.run(self.sess, self.ops_train, currentFeed)       

                    
                    
            loss_currentTrainEpoch += currentLoss_train
            correctPredictNum_currentTrainEpoch[batchIte] = currentCorrectPredcitNum_train

            sequenceLengths_currentTrainEpoch[batchIte] = seqlenBatch_train.sum()#shape[0]#.sum()
                     
        return [loss_currentTrainEpoch / self.batchLoader_train.getBatchNumPerEpoch(), \
                correctPredictNum_currentTrainEpoch.sum() * 1.0 / (self.batchLoader_train.getBatchNumPerEpoch() * self.batchLoader_train.batchSize)]
    
    
    
    def getPrediction_currentTestEpoch(self, batchSize_test, batchSource_test):
        
        self.batchLoader_test = BasicGeneralBatchLoader(batchSize_test, batchSource_test)
        
        
        predictions = list()
        groundTruth = list()
        #self.batchLoader_test.reset()

                

        for batchIte in range(self.batchLoader_test.getBatchNumPerEpoch()):
            [sampleBatch_test, frameLabelBatch_test, seqlenBatch_test] = self.batchLoader_test.getNextBatch()
                
                     
            sampleBatch_test = np.array(sampleBatch_test)
            seqlenBatch_test = np.array(seqlenBatch_test)
                     
                     
            currentFeed = {self.placeHolder_samples: sampleBatch_test, \
                           self.placeHolder_labels: frameLabelBatch_test[:,0], \
                           self.placeHolder_seqLens: seqlenBatch_test}
            
            currentPrediction = self.run(self.sess, self.prediction, currentFeed)
            
            for i in range(self.batchLoader_test.batchSize):
                predictions.append(currentPrediction[i, : seqlenBatch_test[i], :])
                groundTruth.append(frameLabelBatch_test[i][0])
        
        return predictions, groundTruth


    
    def testOneEpoch(self):
        correctPredictNum_currentTestEpoch = np.zeros(self.batchLoader_test.getBatchNumPerEpoch())
        sequenceLengths_currentTestEpoch = np.zeros(self.batchLoader_test.getBatchNumPerEpoch())
        loss_currentTestEpoch = 0.0

        self.batchLoader_test.reset()
        sampleNum = self.batchLoader_test.sampleNum
        
        predictions = np.zeros([self.batchLoader_test.getBatchNumPerEpoch() * self.batchLoader_test.batchSize, self.classNumAll])
        groundTruths = np.zeros(self.batchLoader_test.getBatchNumPerEpoch() * self.batchLoader_test.batchSize)
     

        for batchIte in range(self.batchLoader_test.getBatchNumPerEpoch()):
            [sampleBatch_test, frameLabelBatch_test, seqlenBatch_test] = self.batchLoader_test.getNextBatch()
                
                     
            sampleBatch_test = np.array(sampleBatch_test)
            frameLabelBatch_test = np.array(frameLabelBatch_test)
            seqlenBatch_test = np.array(seqlenBatch_test)
                     
                     
            currentFeed = {self.placeHolder_samples: sampleBatch_test, \
                           self.placeHolder_labels: frameLabelBatch_test[:,0], \
                           self.placeHolder_seqLens: seqlenBatch_test}
            
            currentLoss_test, currentPrediction, currentCorrectPredcitNum_test = self.run(self.sess, self.ops_test, currentFeed)
     
            predictions[batchIte * self.batchLoader_test.batchSize : (batchIte + 1) * self.batchLoader_test.batchSize, :] = currentPrediction
            groundTruths[batchIte * self.batchLoader_test.batchSize : (batchIte + 1) * self.batchLoader_test.batchSize] = frameLabelBatch_test[:,0]
            
            
            loss_currentTestEpoch += currentLoss_test
            correctPredictNum_currentTestEpoch[batchIte] = currentCorrectPredcitNum_test
            sequenceLengths_currentTestEpoch[batchIte] = seqlenBatch_test.sum()#.shape[0]#.sum()
        predictions = predictions[:sampleNum, :]
        groundTruths = groundTruths[:sampleNum]
        predictions = predictions.argmax(1)    
        correctPrediction = predictions == groundTruths
        correctPredictionNum = correctPrediction.sum()
        accuracy = correctPredictionNum * 1.0 / sampleNum                    
        
        return [loss_currentTestEpoch / self.batchLoader_test.getBatchNumPerEpoch(), accuracy]


