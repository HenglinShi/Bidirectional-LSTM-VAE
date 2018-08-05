'''
Created on Jan 12, 2018

@author: hshi
'''

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import variable_scope as vs

def linearLayer(name, bottom, outDim, activation = True):    
    
    with tf.variable_scope(name) as scope:
        
        weight = tf.Variable(tf.random_normal([bottom.shape[-1].value, outDim]), name='weights')
        
        biases = tf.Variable(tf.random_normal([outDim]), name='biases')
        
        if activation:
            return tf.nn.relu(tf.add(tf.matmul(bottom, weight), biases, 'activation'), name=scope.name)
        else:
            return tf.add(tf.matmul(bottom, weight), biases, name=scope.name)
        

def bidirectionalLstmLayer(scopeName, bottom_fw, bottom_bw, outDim, sequence_length=None,
                              initial_state_fw=None, initial_state_bw=None,
                              dtype=None, parallel_iterations=None,
                              swap_memory=False, time_major=False, scope=None):
  
    with vs.variable_scope(scopeName):
        # Forward direction
            
    
        with vs.variable_scope("fw") as fw_scope:
        
            cell_fw = rnn.BasicLSTMCell(outDim, forget_bias=1.0)
            output_fw, output_state_fw = tf.nn.dynamic_rnn(cell=cell_fw, inputs=bottom_fw, sequence_length=sequence_length,
                                                           initial_state=initial_state_fw, dtype=dtype,
                                                           parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                                                           time_major=time_major, scope=fw_scope)

    
        



        with vs.variable_scope("bw") as bw_scope:
            
            cell_bw = rnn.BasicLSTMCell(outDim, forget_bias=1.0)

            output_bw, output_state_bw = tf.nn.dynamic_rnn(cell=cell_bw, inputs=bottom_bw, sequence_length=sequence_length,
                                                     initial_state=initial_state_bw, dtype=dtype,
                                                     parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                                                     time_major=time_major, scope=bw_scope)


        #outputs = (output_fw, output_bw)
        #output_states = (output_state_fw, output_state_bw)

        return output_fw, output_bw, output_state_fw, output_state_bw

