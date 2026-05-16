from __future__ import division

import os
import sys
import time
import numpy as np
import scipy.io as sio
import tensorflow as tf
import math
import json

from network import Model
from enum import Enum

from seq_tools import compute_projection_matrices, compute_covariance
import pickle

runType = Enum('runType', ['Full', 'DSManifPert', 'SSManifPert','ControlManifPert'])


# Set configuration parameters
# these are all default parameters not set in argparse
def getconfig(kwargs):
    config = {'rnn_type'    : 'LeakyRNN',  
              'activation'  : 'softplus',    # relu, softplus, tanh, elu
              'tau'         : 100,           # ms
              'tau_noise'   : 2,             # ms
              'dt'          : 1,             # discretization time step
              'sigma_rec'   : 0.05,          # noise scale
              'w_rec_init'  : 'randortho',   # diag, randortho, randgauss
              'training_iters' : 10000000,   # Max. number of trials to run
              'SAVE_PARAMS' : True           # Whether to save model parameters after each problem
    }

    config['alpha'] = np.float32(1.0 * config['dt'] / config['tau'])               # Discretization - network
    config['alpha_noise'] = np.float32(1.0 * config['dt'] / config['tau_noise'])   # Discretization - noise

    config.update(kwargs)

    # Set random seed
    rng = np.random.RandomState(config['seed'])
    config['rng']           = rng

    config['image_shape'] = [10]
    config['num_input'] = np.prod(config['image_shape']) + 1 #Image + fixation stim
    config['num_rnn'] = 100
    config['num_rnn_out'] = 2 + 1 # Saccades + Fixation
    config['fixationInput'] = 1.0/np.sqrt(np.prod(config['image_shape']))

    # Trial duration parameters
    config['totalLength'] = config['stimLength']+config['delayLength']+config['decisionLength']
    config['tdim'] = int(config['totalLength']/config['dt'])
    config['stimPeriod']   = np.array([0, int(config['stimLength']/config['dt'])])
    config['fixationPeriod']  = np.array([0, int((config['stimLength']+config['delayLength'])/config['dt'])])
    config['decisionPeriod'] = np.array([int((config['stimLength']+config['delayLength'])/config['dt']), int((config['stimLength']+config['delayLength']+config['decisionLength'])/config['dt'])])

    config['manifold_perturbation_total'] = config['max_tasks']-1
    config['manifold_perturbation_threshold'] = int((config['max_tasks']-1)/2)
    
    config['alpha_projection'] = 1e-3
    
    # Display configuration
    for key, val in config.items():
        print('{:20s} = '.format(key) + str(val))
    
    save_config(config)

    return config

def save_config(config):
    # Custom function to handle non-serializable data types
    def custom_serializer(obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to a list
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, np.random.mtrand.RandomState):
            return None  # Exclude the RandomState object from serialization
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    savepath = os.path.join('data', config['save_dir'])
    if not os.path.exists(os.path.join(savepath, 'ckpts')):
        os.makedirs(os.path.join(savepath, 'ckpts'))
        os.makedirs(os.path.join(savepath, 'perfs'))
        os.makedirs(os.path.join(savepath, 'saved'))
        os.makedirs(os.path.join(savepath, 'trIms'))

    with open(os.path.join('data', config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, default=custom_serializer)

# save weights and biases
def testAndSaveParams(sess, config, model, images, taskIndex, suff=''):
    if suff != '':
        suff = '_'+ suff

    # Save images
    dat = dict()
    dat['images'] = images

    # Save weights
    wts, wtNames = model.getWeights(sess)
    for wt, nm in zip(wts, wtNames):
        dat['wts_'+nm] = wt

    # Write to file
    sio.savemat(os.path.join('data', config['save_dir'], 'saved', str(taskIndex) +suff+ '.mat'),dat)


# Simulate and save learned trajectories -
# to infer decision and stimulus subspaces,
# and subsequently perform manifold perturbations
def getStates(sess, config, model, images):

    st0 = dict()
    
    for stim in range(config['num_rnn_out']-1):
        _, trials = generateData(config, images, test=True, stim=[stim])
    
        # Generate feed_dict
        feed_dict = {model.x: trials['x'],
                     model.y_rnn: trials['y_rnn'],
                     model.y_rnn_mask: trials['y_rnn_mask']}
    
        # Test model
        st, op, err = sess.run([model.states, model.y_hat_, model.cost_lsq_rnn], feed_dict=feed_dict)

        st0[stim] = st

    return st0

def generateImages(config):
    images_all = np.empty((config['max_tasks'], config['num_rnn_out']-1, config['image_shape'][0]))
    for k in range(config['max_tasks']):
        # [2, 10]
        images = config['rng'].normal(size=[config['num_rnn_out']-1] + config['image_shape']).astype(np.float32)
        for stim in range(config['num_rnn_out']-1):
            images[stim,:] = images[stim,:]/np.linalg.norm(images[stim,:])
        proj = np.dot(images[0, :], images[1, :])
        images[1, :] -= proj*images[0, :]
        for stim in range(config['num_rnn_out']-1):
            images[stim,:] = images[stim,:]/np.linalg.norm(images[stim,:])
        images_all[k] = images
    return images_all

            
# Generate batch of trials
def generateData(config, images = None, test = False, stim = None):
    # Draw new sample images at random and orthonormalize
    if images is None:
        raise Exception('No images provided during testing')

    # Create input (x) and target (y_rnn) and output temporal mask (y_rnn_mask) matrices
    trials = dict()

    if test:
        stims = np.array(stim)
        datasetSize = 1
    else:
        stims = config['rng'].randint(config['num_rnn_out']-1, size=[config['batch_size']])
        datasetSize = config['batch_size']

    trials['x'] = images[stims,:]
    trials['x'] = np.concatenate((trials['x'], np.float32(config['fixationInput']*np.ones([datasetSize, 1]))), axis=1)

    fixationOffset = int(datasetSize*config['fixationPeriod'][1])
    trials['y_rnn'] = np.zeros((datasetSize*config['tdim'], config['num_rnn_out'])) 
    trials['y_rnn'][0:fixationOffset, config['num_rnn_out']-1] = 1.0 # Fixation
    for stCnt, stim in enumerate(stims):
        trials['y_rnn'][np.arange(fixationOffset+stCnt, datasetSize*config['tdim'], datasetSize), stim] = 1.0
    trials['y_rnn'] = trials['y_rnn'].astype(np.float32)

    tmp = np.ones([datasetSize*config['tdim']])
    tmp[np.arange(fixationOffset, fixationOffset+datasetSize*int(100/config['dt']))] = 0.0
    trials['y_rnn_mask'] = tmp.astype(bool)

    return stims, trials

def test_input(config, sess, model, images, idx):
    # 1st output is stims
    _, trials = generateData(config, images=images, test=True, stim=[idx])

    feed_dict = {model.x: trials['x'],
                 model.y_rnn: trials['y_rnn'],
                 model.y_rnn_mask: trials['y_rnn_mask']}

    c_lsq = sess.run([model.cost_lsq_rnn], feed_dict=feed_dict)

    return c_lsq


# Generate model and train network
def notrain(**kwargs):

    config = getconfig(kwargs)

    t_start = time.time()

    images_all = generateImages(config)
    np.save(os.path.join('data', config['save_dir'], 'images_all.npy'), images_all)

    if config['runType'] != runType.Full: # for manifold perturbation only
        saveStates = np.zeros((config['num_rnn'], config['tdim'], config['manifold_perturbation_total']))

    # Reset tensorflow graphs
    tf.compat.v1.reset_default_graph() 

    # Use customized session that also launches the graph
    with tf.compat.v1.Session() as sess:
        model = Model(config=config) # Generate graph
        model.initialize(sess) # Initialize graph
        model.printTrainable() # List trainable vars
        #sess.graph.finalize() # can't do this if graph is altered during training

        convCnt = []
        perf = []
        trIm = []
        wNormR2 = []
        wNormR = []
        wNormI = []
        wNormO = []
        hNorm = []
        HM = []
        singVals = np.zeros([config['max_tasks'],100])
        images = images_all[0]
        firstConv = False
       
        test_perfs = {}
        for k in range(config['max_tasks']):
            test_perfs[k] = []
        test_perfs['trials'] = []
        test_perfs['training_task'] = []

        for trial in range(config['training_iters']):
            # test on all images
            if (trial % config['trialsPerTest'] == 0) or (images is None):
                for k, testImages in enumerate(images_all):
                    c_lsq0 = test_input(config, sess, model, testImages, 0)
                    c_lsq1 = test_input(config, sess, model, testImages, 1)
                    test_perfs[k].append((c_lsq0[0], c_lsq1[0]))
                test_perfs['trials'].append(trial)
                test_perfs['training_task'].append(len(convCnt))

            # Generate a batch of trials
            images = images_all[len(convCnt)]
            stims, trials = generateData(config, images)
            print(trials)
            trIm.extend(stims.tolist())

            # Generate feed_dict
            feed_dict = {model.x: trials['x'],
                         model.y_rnn: trials['y_rnn'],
                         model.y_rnn_mask: trials['y_rnn_mask']}

            # Run forward + backward passes
            c_lsq = sess.run([model.cost_lsq_rnn], feed_dict=feed_dict)[0]

            # Save trial specific learning stats
            perf.append(c_lsq)

            # Print summary stats
            runTime = time.time()-t_start
            if trial%100 == 0:
                print('Trial: ' + str(trial) + ' cost: ' + str(np.mean(perf[-50:])) + ' cost_lsq: ' + str(c_lsq) + ' Runtime: ' + str(runTime) + ' s')
                sys.stdout.flush()

            # Check for convergence, set converged flagW, save model
            if math.isnan(c_lsq) or len(perf) > 20000:
                taskFailed = True
            else:
                taskFailed = False
            
            # Saved trained model for new problem
            condition = (len(perf) > 50) or taskFailed
            if condition:
                if taskFailed: # Update problem learning-specific stats when convergence fails
                    convCnt.append(np.nan)
                    wNormR.append(np.nan)
                    wNormR2.append(np.nan)
                    wNormI.append(np.nan)
                    wNormO.append(np.nan)
                    hNorm.append(np.nan)
                    HM.append(np.nan)
                else:
                    convCnt.append(len(perf)-50) # Trials to convergence for new problem

                    # Dump trained model and problem specifics to file after it is learned
                    if len(convCnt) >= 1 and config['SAVE_PARAMS'] == True:
                        # mat file
                        testAndSaveParams(sess, config, model, images, len(convCnt))
                        # test_perfs.pkl
                        if config['replace_test_perfs']:
                            with open(os.path.join('data', config['save_dir'], 'perfs.pkl'),'wb') as file:
                                pickle.dump(test_perfs, file)
                        else:
                            with open(os.path.join('data', config['save_dir'], 'perfs', str(len(convCnt)) + '.pkl'),'wb') as file:
                                pickle.dump(test_perfs, file)

                    # Save problem learning-specific stat summary
                    np.savetxt(os.path.join('data', config['save_dir'], 'trIms', str(len(convCnt)) + '.txt'), np.array(trIm), fmt='%f', delimiter=' ')

                    # # Set firing rate homeostatic set point after first problem is learned
                    # if firstConv == False:
                    #     firstConv = True
                    #     model.updateRegularizerTargets(hNorm[-1], wNormR[-1], wNormI[-1], sess)

                    if config['runType'] != runType.Full:  # for manifold perturbation only
                        if len(convCnt) == 1:
                            model.save(len(convCnt))
                    currSingVals = sess.run([model.sings])
                    singVals[len(convCnt)-1,:] = currSingVals[0]

                    if config['runType'] == runType.Full:
                        if len(convCnt) <= 30:
                            model.save(len(convCnt))

                    # 2023-08-01
                    # Write training summaries to file
                    with open(os.path.join('data', config['save_dir'], 'conv.txt'), 'a') as f:
                        # tested and works on np.nan
                        f.write(f'{convCnt[-1]:f}\n')
                    with open(os.path.join('data', config['save_dir'], 'SINGS.txt'), 'a') as f:
                        f.write('  '.join([f'{val:12.9f}' for val in singVals[len(convCnt)-1]]) + '\n')
                
                sys.stdout.flush()


                # Reset problem specific stats for new problem
                perf = []
                wNR = []
                wNR2 = []
                wNI = []
                wNO = []
                hN = []
                hm = []
                trIm = []
                images = None # This initiates sampling of new images for next problem
                
                # Reset adam's internals before onset of learning new problem
                model.resetOpt(sess)

            # Done learning all problems?
            if len(convCnt) >= config['max_tasks']:
                break

    print(convCnt)