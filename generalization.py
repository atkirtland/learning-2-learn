from __future__ import division

import os
import numpy as np
import tensorflow as tf

from network import Model

from train import runType
from train import get_defaultconfig
import scipy.io
from train import test_input

def save_list_to_file(lst, file_path):
    with open(file_path, 'a') as file:
        file.write(str(lst) + '\n')

# Generate model and train network
def test_generalization(seed          = 0,
          batchSize     = 1,
          l2            = 0.0005,
          l2_wR          = 0.001,
          l2_wI          = 0.0001,
          l2_wO          = 0.1,
          learningRateInit = 0.001,
          beta1 = 0.9,
          beta2 = 0.999,
          svBnd = 10.0,
          rType = runType.Full,
          projGrad = True,
          originalAdam = False,
          **kwargs):

    save_name = '{:d}_{:d}_{:d}_{:f}_{:f}_{:f}_{:s}_matrices-test'.format(seed, projGrad, originalAdam, learningRateInit, beta1, beta2, rType)
                                    
    # Set random seed
    rng = np.random.RandomState(seed)

    # Setup hyper-parameters
    config = get_defaultconfig()
    config['seed']          = seed
    config['batch_size']    = batchSize
    config['rng']           = rng
    config['save_name']     = save_name
    config['l2_h'] = l2
    config['l2_wR'] = l2_wR
    config['svBnd'] = svBnd
    config['l2_wI'] = l2_wI
    config['l2_wO'] = l2_wO

    config['init_lr_full'] = learningRateInit
    config['beta1'] = beta1
    config['beta2'] = beta2

    # Allow for additional configuration options
    for key, val in kwargs.items():
        config[key] = val

    config['image_shape'] = [10]
    config['num_input'] = np.prod(config['image_shape']) + 1 #Image + fixation stim
    config['num_rnn'] = 100
    config['num_rnn_out'] = 2 + 1 # Saccades + Fixation
    config['fixationInput'] = 1.0/np.sqrt(np.prod(config['image_shape']))


    # Trial duration parameters
    config['tdim'] = int(2000/config['dt'])
    config['stimPeriod']   = np.array([0, int(500/config['dt'])])
    config['fixationPeriod']  = np.array([0, int(1500/config['dt'])])
    config['decisionPeriod'] = np.array([int(1500/config['dt']), int(2000/config['dt'])])

    config['runType'] = rType
    config['max_tasks'] = 1001
    if config['runType'] != runType.Full:
        config['max_tasks'] = 101

    config['alpha_projection'] = 1e-3
    config['projGrad'] = projGrad
    config['originalAdam'] = originalAdam

    # Display configuration
    for key, val in config.items():
        print('{:20s} = '.format(key) + str(val))

    # Reset tensorflow graphs
    tf.compat.v1.reset_default_graph() 

    # Use customized session that also launches the graph
    with tf.compat.v1.Session() as sess:
        model = Model(config=config) # Generate graph
        model.initialize(sess) # Initialize graph
        model.printTrainable() # List trainable vars

        sCnt = 30
        model.restore(sCnt)

        test_cl = False
        if test_cl:
            # go through past images
            perfs = []
            for k in range(30):
                # from testAndSaveParams
                mat = scipy.io.loadmat(os.path.join('data', 'saved_' + config['save_name'] + '_' + str(k+1) +''+ '.mat'))
                images = mat['images']
                c_lsq0 = test_input(config, sess, model, images, 0)
                c_lsq1 = test_input(config, sess, model, images, 1)
                perfs.append((c_lsq0[0]+c_lsq1[0])/2)
            print(perfs)
        else:
            # systematically add noise to last image
            mat = scipy.io.loadmat(os.path.join('data', 'saved_' + config['save_name'] + '_' + str(30) +''+ '.mat'))
            images = mat['images']
            perfs = []
            # for scale in np.linspace(0.0, 2.0, num=20):
            for scale in np.logspace(-3, 1, num=25):
                avg = 0
                N_trials = 10
                for _ in range(N_trials):
                    noise = config['rng'].normal(loc=0.0, scale=scale, size=[config['num_rnn_out']-1] + config['image_shape']).astype(np.float32)
                    images2 = images + noise
                    c_lsq0 = test_input(config, sess, model, images2, 0)
                    c_lsq1 = test_input(config, sess, model, images2, 1)
                    avg += (c_lsq0[0]+c_lsq1[0])/2
                perfs.append(avg/N_trials)
            save_list_to_file(perfs, "sysgen.txt")
    print("done!")

for seed in [0, 1]:
    for projGrad in [True, False]:
        test_generalization(seed          = 0,
            batchSize     = 1,
            l2            = 0.0005,
            l2_wR          = 0.001,
            l2_wI          = 0.0001,
            l2_wO          = 0.1,
            svBnd = 10.0,
            rType = runType.Full,
            projGrad = projGrad,
            learningRateInit = 0.001,
            beta1 = 0.9,
            beta2 = 0.999)