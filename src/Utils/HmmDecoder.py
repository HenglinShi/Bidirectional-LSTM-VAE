'''
Created on Jan 12, 2018

@author: hshi
'''
import numpy as np

class HmmDecoder(object):
    '''
    classdocs
    '''


    def __init__(self, stateSequenceList = None, stateNum = None, prior = None, transitionMatrix = None):
        '''
        Constructor
        '''
        self.stateSequenceList = stateSequenceList
        self.stateNum = stateNum
        self.prior = prior
        self.transitionMatrix = transitionMatrix
        
        self.emissionScalar = None
        
        if self.prior is None or self.transitionMatrix is None:
            self.initPriors()
        
    def initPriors(self):
        
           
        self.prior = np.zeros(self.stateNum)
        
        self.transitionMatrix = np.zeros((self.stateNum, self.stateNum))
        
        self.emissionScalar = np.zeros((self.stateNum))
        #transitionScalar = np.zeros(self.stateNum)
    
        
    
        for i in range(len(self.stateSequenceList)):
            
            currentStateSequence = self.stateSequenceList[i]
            self.prior[int(currentStateSequence[0])] += 1
            
            for j in range(currentStateSequence.shape[0] - 1):
                self.transitionMatrix[int(currentStateSequence[j]), int(currentStateSequence[j + 1])] += 1
                
                #transitionScalar[:, int(currentStateSequence[j])] += 1
                
                self.emissionScalar[int(currentStateSequence[j])] += 1
            
            self.emissionScalar[int(currentStateSequence[-1])] += 1
            

        #self.transitionMatrix = np.log(np.divide(self.transitionMatrix, transitionScalar))
        self.prior = np.log(self.prior * 1.0 / len(self.stateSequenceList))
        self.emissionScalar = np.log(self.emissionScalar * 1.0 / self.emissionScalar.sum())
        self.transitionMatrix = np.log(self.transitionMatrix * 1.0 / self.transitionMatrix.sum(1).reshape(-1, 1))
        
    def decode(self, emissionMatrix):
            
        emissionMatrix = emissionMatrix - self.emissionScalar
        emissionMatrix = emissionMatrix.transpose()   
            
        T = emissionMatrix.shape[-1]
        N = emissionMatrix.shape[0]
        
        path = np.zeros(T, dtype=np.int32)
        global_score = np.zeros(shape=(N,T))
        predecessor_state_index = np.zeros(shape=(N,T), dtype=np.int32)
    
        t = 1
        
        #tmp = prior + observ_likelihood[:, 0]
        #global_score[:, 0] = tmp[:, 0]
        if len(self.prior.shape) == 1:
            global_score[:, 0] = self.prior + emissionMatrix[:, 0]
        else:
            global_score[:, 0] = self.prior[:, 0] + emissionMatrix[:, 0]
        
        #global_score[:, 0] = prior + observ_likelihood[:, 0]
        # need to  normalize the data
        
        for t in range(1, T):
            for j in range(N):
                temp = global_score[:, t-1] + self.transitionMatrix[:, j] + emissionMatrix[j, t]
                global_score[j, t] = max(temp)
                predecessor_state_index[j, t] = temp.argmax()
    
        path[T-1] = global_score[:, T-1].argmax()
        
        for t in range(T-2, -1, -1):
            path[t] = predecessor_state_index[ path[t+1], t+1]
    
        return [path, predecessor_state_index, global_score]
        
    
    
    
    
    
    