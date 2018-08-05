'''
Created on Jan 12, 2018

@author: hshi
'''
from Model.BasicModels import bidirectionalLstmNetwork

import tensorflow as tf
from Model.BasicOps import linearLayer


def oneLayerBidirectionalLstmEncoder(bottom, sequenceLengths, hiddenDims_1, hiddenDims_z):
    time_dim = 1
    batch_dim = 0
    
    batchSize = bottom.shape[0].value
    maxSequLen = bottom.shape[1].value
    
    
    samples_bw = tf.reverse_sequence(bottom, sequenceLengths, time_dim, batch_dim, 'vae_encoder_reverse_bottom')
    output_fw, output_bw, _, _ = bidirectionalLstmNetwork('vae_encoder_BidirectionalLstm', bottom, samples_bw, 1, [hiddenDims_1], sequenceLengths)

    
    output_bw_reversed = tf.reverse_sequence(output_bw, sequenceLengths, time_dim, batch_dim, 'vae_encoder_reverse_blstm_out')
    biLstmOut1 = tf.concat([output_fw, output_bw_reversed], 2)
    biLstmOut = tf.reshape(biLstmOut1, [-1, 2 * hiddenDims_1])

    linear1 = linearLayer('vae_encoder_FC_1', biLstmOut, hiddenDims_1, True)
    
    
    mean = tf.reshape(linearLayer("vae_encoder_mu", linear1, hiddenDims_z, False), shape = [-1, maxSequLen, hiddenDims_z])
    SE_ln = tf.reshape(linearLayer("vae_encoder_SE_log", linear1, hiddenDims_z, False), shape = [-1, maxSequLen, hiddenDims_z])
    
    return mean, SE_ln

def vaeSampler_forLstm(bottom_mean, bottom_SE_ln, bottom_noise):
    hiddenDim_z = bottom_mean.shape[-1].value
    maxSeqLen = bottom_mean.shape[1].value
    
    mean_flatten = tf.reshape(bottom_mean, shape = [-1, hiddenDim_z])
    SE_ln_flatten = tf.reshape(bottom_SE_ln, shape = [-1, hiddenDim_z])
    noise_flatten = tf.reshape(bottom_noise, shape = [-1, hiddenDim_z])
    
    
    sigma = tf.exp(tf.multiply(0.5, SE_ln_flatten), name = "vae_sampler_sigma")
    SE_sampled = tf.multiply(sigma, noise_flatten, name = "vae_sampler_SE_sampled")
    z = tf.add(SE_sampled, mean_flatten, name = "vae_sampler_z")
    
    z_reshaped = tf.reshape(z, shape = [-1, maxSeqLen, hiddenDim_z])
    
    return z_reshaped

def oneLayerBidirectionalLstmDecoder(bottom, sequenceLengths, hiddenDim_1, hiddenDim_target):
    
    time_dim = 1
    batch_dim = 0
    
    batchSize = bottom.shape[0].value
    maxSeqLen = bottom.shape[1].value
    
    samples_bw = tf.reverse_sequence(bottom, sequenceLengths, time_dim, batch_dim, 'vae_decoder_reverse_bottom')
    output_fw, output_bw, _, _ = bidirectionalLstmNetwork('vae_decoder_BidirectionalLstm', bottom, samples_bw, 1, [hiddenDim_1], sequenceLengths)
    output_bw_reversed = tf.reverse_sequence(output_bw, sequenceLengths, time_dim, batch_dim, 'vae_decoder_reverse_blstm_out')
    
    biLstmOut1 = tf.concat([output_fw, output_bw_reversed], 2)
    
    biLstmOut = tf.reshape(biLstmOut1, [-1, 2 * hiddenDim_1])

    linear_decoder_1 = linearLayer('vae_decoder_FC_1', biLstmOut, hiddenDim_1, True)
    
    reconstruction = tf.reshape(linearLayer("vae_decoder_reconstruction", linear_decoder_1, hiddenDim_target, False), shape = [-1, maxSeqLen, hiddenDim_target])
    
    return reconstruction

def twoLayerFcEncoder_norm(bottom, hiddenDims_1, hiddenDims_z):
    
    linear1 = linearLayer("vae_encoder_1", bottom, hiddenDims_1, True)
    
    linear2 = linearLayer("vae_encoder_mu", linear1, hiddenDims_z, False)

    
    return linear2

def twoLayerFcEncoder(bottom, hiddenDims_1, hiddenDims_z):
    
    linear1 = linearLayer("vae_encoder_1", bottom, hiddenDims_1, True)
    
    mean = linearLayer("vae_encoder_mu", linear1, hiddenDims_z, False)
    SE_ln = linearLayer("vae_encoder_SE_log", linear1, hiddenDims_z, False)
    
    return mean, SE_ln

def vaeSampler(bottom_mean, bottom_SE_ln, bottom_noise):
    
    sigma = tf.exp(tf.multiply(0.5, bottom_SE_ln), name = "vae_sampler_sigma")
    SE_sampled = tf.multiply(sigma, bottom_noise, name = "vae_sampler_SE_sampled")
    z = tf.add(SE_sampled, bottom_mean, name = "vae_sampler_z")
    
    return z



def twoLayerFcDecoder(bottom, hiddenDim_1, hiddenDim_target):
    
    linear_decoder_1 = linearLayer("vae_decoder_1", bottom, hiddenDim_1, True)
    
    reconstruction = linearLayer("vae_decoder_reconstruction", linear_decoder_1, 
                                 hiddenDim_target, False)
    
    return reconstruction
    
    
    

    

def vaeLoss_forLstm(reconstruction, target, mean_squared, SE, SE_ln):
    
    reconstruction_target_dim = reconstruction.shape[-1].value
    maxSeqLen = reconstruction.shape[1].value
    
    hiddenDim_z = mean_squared.shape[-1].value
    
    target_reconstruction_flatten = tf.reshape(target, shape = [-1, reconstruction_target_dim])
    reconstruction_flatten = tf.reshape(reconstruction, shape = [-1, reconstruction_target_dim])
    
    mean_squared_flatten = tf.reshape(mean_squared, shape = [-1, hiddenDim_z]) 
    SE_flatten = tf.reshape(SE, shape = [-1, hiddenDim_z]) 
    SE_ln_flatteb = tf.reshape(SE_ln, shape = [-1, hiddenDim_z]) 
    
    KLD = tf.reduce_sum(tf.add(-0.5, 
                               0.5 * mean_squared_flatten + \
                               0.5 * SE_flatten - \
                               0.5 * SE_ln_flatteb), 
                        reduction_indices = 1,
                        name = "vae_loss_KLD")
    

    BCE = tf.reduce_sum(tf.square(tf.subtract(reconstruction_flatten, target_reconstruction_flatten)), reduction_indices=1, name = "vae_loss_BCE")
    
    return tf.add(KLD, BCE * 0.5, "vae_loss_overall"), KLD, BCE * 0.5

def vaeLoss_forLstm1(reconstruction, target, mean_squared, SE, SE_ln):
    
    KLD = tf.reduce_sum(tf.add(-0.5, 0.5 * mean_squared + 0.5 * SE - 0.5 * SE_ln), reduction_indices = 2, name = "vae_loss_KLD")
    

    BCE = tf.reduce_sum(tf.square(tf.subtract(reconstruction, target)), reduction_indices=2, name = "vae_loss_BCE")
    
    return tf.add(KLD, BCE * 0.5, "vae_loss_overall"), KLD, BCE * 0.5

def normalEncoderLoss(reconstruction, target):
    
    #KLD = tf.reduce_sum(tf.add(-0.5, 0.5 * mean_squared + 0.5 * SE - 0.5 * SE_ln), reduction_indices = 2, name = "vae_loss_KLD")
    

    BCE = tf.reduce_sum(tf.square(tf.subtract(reconstruction, target)), reduction_indices=2, name = "vae_loss_BCE")
    
    return BCE * 0.5
    
def vaeLoss(reconstruction, target, mean_squared, SE, SE_ln):
        
    
    KLD = tf.reduce_sum(tf.add(-0.5, 
                               0.5 * mean_squared + \
                               0.5 * SE - \
                               0.5 * SE_ln), 
                        reduction_indices = 1,
                        name = "vae_loss_KLD")
    

    BCE = tf.reduce_sum(tf.square(tf.subtract(reconstruction, target)), reduction_indices=1, name = "vae_loss_BCE")
    
    return tf.add(KLD, BCE * 0.5, "vae_loss_overall"), KLD, BCE * 0.5


def linearLayer(name, bottom, kernelSize, withActivation=False):
    
    with tf.variable_scope(name) as scope:

        weight = tf.Variable(tf.truncated_normal(shape = [bottom.shape[-1].value, kernelSize],
                                                 stddev=0.001), name='weights')
        
        biases = tf.Variable(tf.constant(0., shape = [kernelSize]), name='biases')
        
        if withActivation:
            return tf.nn.relu(tf.add(tf.matmul(bottom, weight), biases, 'activation'), name=scope.name)
        else:
            return tf.add(tf.matmul(bottom, weight), biases, name=scope.name)

def blstmFramePredictionNetwork(samples, seqenceLengths, sequenceLength_max, outputNum, layerNum = 3, neuronNum = [512]):
    
    time_dim = 1
    batch_dim = 0
    
    samples_bw = tf.reverse_sequence(samples, seqenceLengths, time_dim, batch_dim, 'reverse_bottom')
    output_fw, output_bw, _, _ = bidirectionalLstmNetwork('BidirectionalLstm', samples, samples_bw, layerNum, neuronNum, seqenceLengths)
    output_bw_reversed = tf.reverse_sequence(output_bw, seqenceLengths, time_dim, batch_dim, 'reverse_blstm_out')
    
    biLstmOut1 = tf.concat([output_fw, output_bw_reversed], 2)
    biLstmOut = tf.reshape(biLstmOut1, [-1, 2 * neuronNum[-1]])

    
    fc1 = linearLayer('FC_1', biLstmOut, neuronNum[-1], True)
    
    fc1_dropout = tf.nn.dropout(fc1, 0.5)
    
    prediction = tf.reshape(linearLayer('Linear_1', fc1_dropout, outputNum, False), [-1, sequenceLength_max, outputNum])
    
    
    return prediction


def blstmSequencePredictionNetwork(samples, seqenceLengths, sequenceLength_max, outputNum, layerNum = 3, neuronNum = [512]):
    
    time_dim = 1
    batch_dim = 0
    
    samples_bw = tf.reverse_sequence(samples, seqenceLengths, time_dim, batch_dim, 'reverse_bottom')
    output_fw, output_bw, _, _ = bidirectionalLstmNetwork('BidirectionalLstm', samples, samples_bw, layerNum, neuronNum, seqenceLengths)
    output_bw_reversed = tf.reverse_sequence(output_bw, seqenceLengths, time_dim, batch_dim, 'reverse_blstm_out')
    
    biLstmOut1 = tf.concat([output_fw, output_bw_reversed], 2)
    

    batch_size = tf.cast(tf.shape(biLstmOut1)[0], dtype=tf.int64)
    # Start indices for each sample
    index = tf.range(0, batch_size) * sequenceLength_max + (seqenceLengths - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(biLstmOut1, [-1, 2 * neuronNum[-1]]), index)
    
    fc1 = linearLayer('FC_1', outputs, neuronNum[-1], True)
    
    fc1_dropout = tf.nn.dropout(fc1, 0.5)
    
    prediction = linearLayer('Linear_1', fc1_dropout, outputNum, False)
    
    
    return prediction



def myModel3(samples, seqenceLengths, sequenceLength_max, outputNum):
    
    
    time_dim = 1
    batch_dim = 0
    n_classes = 20
    
    
    samples_bw = tf.reverse_sequence(samples, seqenceLengths, time_dim, batch_dim, 'reverse_bottom')
    output_fw, output_bw, _, _ = bidirectionalLstmNetwork('BidirectionalLstm', samples, samples_bw, 3, [512], seqenceLengths)
    output_bw_reversed = tf.reverse_sequence(output_bw, seqenceLengths, time_dim, batch_dim, 'reverse_blstm_out')
    
    biLstmOut1 = tf.concat([output_fw, output_bw_reversed], 2)
    biLstmOut = tf.reshape(biLstmOut1, [-1, 2 * 512])

    
    fc1 = linearLayer('FC_1', biLstmOut, 512, True)
    
    fc1_dropout = tf.nn.dropout(fc1, 0.5)
    
    prediction = linearLayer('Linear_1', fc1_dropout, outputNum, False)
    
    
    return prediction

def myModel4(samples, seqenceLengths, sequenceLength_max, outputNum):
    
    
    time_dim = 1
    batch_dim = 0
    n_classes = 20
    samples_bw = tf.reverse_sequence(samples, seqenceLengths, time_dim, batch_dim, 'reverse_bottom')
    
    
    
    output_fw, output_bw, _, _ = bidirectionalLstmNetwork('BidirectionalLstm', samples, samples_bw, 4, [1024], seqenceLengths)

    
    output_bw_reversed = tf.reverse_sequence(output_bw, seqenceLengths, time_dim, batch_dim, 'reverse_blstm_out')
    
    biLstmOut1 = tf.concat([output_fw, output_bw_reversed], 2)
    
    biLstmOut = tf.reshape(biLstmOut1, [-1, 2 * 1024])

    
    fc1 = linearLayer('FC_1', biLstmOut, 1024, True)
    
    fc1_dropout = tf.nn.dropout(fc1, 0.5)
    
    prediction = linearLayer('Linear_1', fc1_dropout, outputNum, False)
    
    
    return prediction

def myModel(samples, seqenceLengths, sequenceLength_max, outputNum):
    
    
    time_dim = 1
    batch_dim = 0
    n_classes = 20
    samples_bw = tf.reverse_sequence(samples, seqenceLengths, time_dim, batch_dim, 'reverse_bottom')
    
    
    
    output_fw, output_bw, _, _ = bidirectionalLstmNetwork('BidirectionalLstm', samples, samples_bw, 3, [1024], seqenceLengths)

    
    output_bw_reversed = tf.reverse_sequence(output_bw, seqenceLengths, time_dim, batch_dim, 'reverse_blstm_out')
    
    biLstmOut1 = tf.concat([output_fw, output_bw_reversed], 2)
    
    biLstmOut = tf.reshape(biLstmOut1, [-1, 2 * 1024])

    
    fc1 = linearLayer('FC_1', biLstmOut, 1024, True)
    
    fc1_dropout = tf.nn.dropout(fc1, 0.5)
    
    prediction = linearLayer('Linear_1', fc1_dropout, outputNum, False)
    
    
    return prediction

def myModel2(samples, seqenceLengths, sequenceLength_max, outputNum):
    
    
    time_dim = 1
    batch_dim = 0
    n_classes = 20
    samples_bw = tf.reverse_sequence(samples, seqenceLengths, time_dim, batch_dim, 'reverse_bottom')
    
    
    
    output_fw, output_bw, _, _ = bidirectionalLstmNetwork('BidirectionalLstm', samples, samples_bw, 1, [1024], seqenceLengths)

    
    output_bw_reversed = tf.reverse_sequence(output_bw, seqenceLengths, time_dim, batch_dim, 'reverse_blstm_out')
    
    biLstmOut1 = tf.concat([output_fw, output_bw_reversed], 2)
    
    biLstmOut = tf.reshape(biLstmOut1, [-1, 2 * 1024])

    
    fc1 = linearLayer('FC_1', biLstmOut, 1024, True)
    
    fc1_dropout = tf.nn.dropout(fc1, 0.5)
    
    prediction = linearLayer('Linear_1', fc1_dropout, outputNum, False)
    
    
    return prediction