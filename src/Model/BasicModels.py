'''
Created on Jan 12, 2018

@author: hshi
'''
import numpy as np
import tensorflow as tf
from Model.BasicOps import bidirectionalLstmLayer


def bidirectionalLstmNetwork(name, bottom_fw, bottom_bw, layerNum, outDim, sequenceLength):
    

    
    
    if len(outDim) == 1:
        tmp = np.zeros(layerNum)
        tmp[:] = outDim[0]
        outDim = tmp
        
    elif len(outDim) == layerNum:
        pass
    
    else:
        #error
        pass
        
    
        
    for layerIte in range(layerNum):
        scopeName = name + str(layerIte + 1)
        
        bottom_fw, bottom_bw, output_state_fw, output_state_bw = bidirectionalLstmLayer(scopeName, bottom_fw, bottom_bw, 
                                                                                        outDim[layerIte], sequenceLength, dtype=tf.float32)
        
    return bottom_fw, bottom_bw, output_state_fw, output_state_bw