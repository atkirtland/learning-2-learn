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
    config['tdim'] = int(2000/config['dt'])
    config['stimPeriod']   = np.array([0, int(500/config['dt'])])
    config['fixationPeriod']  = np.array([0, int(1500/config['dt'])])
    config['decisionPeriod'] = np.array([int(1500/config['dt']), int(2000/config['dt'])])

    if config['max_tasks'] is None:
        config['max_tasks'] = 1001
        if config['runType'] != runType.Full:
            config['max_tasks'] = 101
    
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
    if not os.path.exists(savepath):
        os.makedirs(savepath)

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
    sio.savemat(os.path.join('data', config['save_dir'], 'saved_'+str(taskIndex) +suff+ '.mat'),dat)


# core function for manifold perturbations
def resetOutWeightsWithSameStruct(sess, model, config, k, rType, eigs = None, eigsS = None, states = None):
    outWts = model.getOutWeights(sess)

    if eigs is None:
        states1 = states[:,:,0:50]
        states2 = states[:,:,50:100]

        # Find decision and stimulus subspaces
        states1Mn = np.mean(states1, axis = 2, keepdims = False)
        states2Mn = np.mean(states2, axis = 2, keepdims = False)

        # print(str(states1Mn.shape) + ' ' + str(states2Mn.shape))
        x = np.concatenate((states1Mn, states2Mn), axis=1)
        # print(str(x.shape))
        U,S,Vh = np.linalg.svd(x, full_matrices=False)
        eigs = np.real(U[:,0:k])

        states1D = np.matmul(np.matmul(eigs, eigs.T), np.reshape(states1,(config['num_rnn'],-1)))
        states2D = np.matmul(np.matmul(eigs, eigs.T), np.reshape(states2,(config['num_rnn'],-1)))
        states1S = np.reshape(states1,(config['num_rnn'],-1))-states1D
        states2S = np.reshape(states2,(config['num_rnn'],-1))-states2D
        xS = np.concatenate((states1S, states2S), axis=1)
        US,SS,VhS = np.linalg.svd(xS, full_matrices=False)
        eigsS = np.real(US)
        
    # Swap out first few PCs of source subspace from output weights
    # with first few PCs of target subspace
    newOutWts = np.copy(outWts)
    newOrder = config['rng'].permutation(np.arange(k))
    for i in range(0,k):
        if rType == runType.SSManifPert:
            # S -> S mainfold perturbation
            oldVec = eigsS[:,i]
            newVec = eigsS[:,newOrder[i]]
            ov = np.dot(outWts.T, oldVec)
            newOutWts = newOutWts - np.outer(oldVec, ov) + np.outer(newVec, ov)

        elif rType == runType.DSManifPert:
            # D -> S mainfold perturbation
            oldVec = eigs[:,i]
            newVec = eigsS[:,newOrder[i]]
            ov = np.dot(outWts.T, oldVec)
            newOutWts = newOutWts - np.outer(oldVec, ov) + np.outer(newVec, ov)

        elif rType == runType.ControlManifPert:
            # Control with frozen weights only
            continue
        
        else:
            raise NotImplementedError()

    model.setOutWeights(sess, newOutWts) # update new output weights
    return outWts, eigs, eigsS

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
    # 3rd output is images
    _, trials = generateData(config, images=images, test=True, stim=[idx])

    feed_dict = {model.x: trials['x'],
                 model.y_rnn: trials['y_rnn'],
                 model.y_rnn_mask: trials['y_rnn_mask']}

    c_lsq = sess.run([model.cost_lsq_rnn], feed_dict=feed_dict)

    return c_lsq


# Generate model and train network
def train(**kwargs):

    config = getconfig(kwargs)

    t_start = time.time()

    images_all = generateImages(config)
    np.save(os.path.join('data', config['save_dir'], 'images_all.npy'), images_all)

    if config['runType'] != runType.Full: # for manifold perturbation only
        saveStates = np.zeros((config['num_rnn'], config['tdim'], 100))

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
        wNR2 = []
        wNR = []
        wNI = []
        wNO = []
        hN = []
        wNormR2 = []
        wNormR = []
        wNormI = []
        wNormO = []
        hNorm = []
        hm = []
        HM = []
        singVals = np.zeros([config['max_tasks'],100])
        images = images_all[0]
        firstConv = False

        if config['projGrad']:
            input_proj = tf.zeros((config['num_rnn'] + config['num_input'], config['num_rnn'] + config['num_input']))
            activity_proj = tf.zeros((config['num_rnn'], config['num_rnn']))
            output_proj = tf.zeros((config['num_rnn_out'], config['num_rnn_out']))
            recurrent_proj = tf.zeros((config['num_rnn'], config['num_rnn']))
        
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

            # set optimizer for each new task
            if config['projGrad'] and (images is None):
                model.buildOpt(config, activity_proj=activity_proj, input_proj=input_proj, output_proj=output_proj, recurrent_proj=recurrent_proj, taskNumber=len(convCnt))
            
            if config['debug_timing']:
                print("buildOpt", time.time()-t_start)

            # Generate a batch of trials
            images = images_all[len(convCnt)]
            stims, trials = generateData(config, images)
            trIm.extend(stims.tolist())

            if config['debug_timing']:
                print("generateData", time.time()-t_start)

            # Generate feed_dict
            feed_dict = {model.x: trials['x'],
                         model.y_rnn: trials['y_rnn'],
                         model.y_rnn_mask: trials['y_rnn_mask']}

            # Run forward + backward passes
            _, c_lsq, c_reg, wnR, wnR2, wnI, wnO, hn, hMax, out, maxSingVal, topTenSings = sess.run([model.optimizer_full, model.cost_lsq_rnn, model.cost_reg_rnn, model.wNormR, model.wNormR2, model.wNormI, model.wNormO, model.hNorm, model.hMax, model.y_hat, model.maxSingVal, model.topTenSings], feed_dict=feed_dict)

            if config['debug_timing']:
                print("sess.run", time.time()-t_start)

            # Save trial specific learning stats
            perf.append(c_lsq)
            wNR.append(wnR)
            wNR2.append(wnR2)
            wNI.append(wnI)
            wNO.append(wnO)
            hN.append(hn)
            hm.append(hMax)

            # Print summary stats
            runTime = time.time()-t_start
            if trial%100 == 0:
                print('Trial: ' + str(trial) + ' cost: ' + str(np.mean(perf[-50:])) + ' cost_reg: ' + str(c_reg) + ' cost_lsq: ' + str(c_lsq) + ' Runtime: ' + str(runTime) + ' s')
                sys.stdout.flush()

            # Check for convergence, set converged flagW, save model
            if math.isnan(c_lsq) or len(perf) > 20000:
                taskFailed = True
            else:
                taskFailed = False
            
            if config['debug_timing']:
                print("next bit", time.time()-t_start)

            # Saved trained model for new problem

            if config['overtraining']:
                condition = (len(perf) >= config['trialsPerTask']) or taskFailed
            else:
                condition = (len(perf) > 50 and np.mean(perf[-50:]) < 0.005) or taskFailed
            if condition:
                if config['projGrad']:
                    # Generate a batch of trials
                    stims, trials = generateData(config, images)

                    # Generate feed_dict
                    feed_dict = {model.x: trials['x'],
                                model.y_rnn: trials['y_rnn'],
                                model.y_rnn_mask: trials['y_rnn_mask']}

                    eval_h, eval_x, eval_y, Win, Wrec = sess.run([model.states, model.x, model.y_rnn, model.w_in, model.w_rec], feed_dict=feed_dict)
                    eval_h = np.expand_dims(eval_h, axis=1)
                    eval_x = np.tile(eval_x, (2000,1,1))
                    eval_y = np.expand_dims(eval_y, axis=1)

                    full_state = np.concatenate([eval_x, eval_h], -1)
                    Wfull = np.concatenate([Win, Wrec], 0)

                    if config['debug_timing']:
                        print("generation", time.time()-t_start)

                    # joint covariance matrix of input and activity
                    Shx_task = compute_covariance(np.reshape(full_state, (-1, config['num_rnn'] + config['num_input'])).T)
                    if config['debug_timing']:
                        print("Shx_task", time.time()-t_start)

                    # covariance matrix of output
                    Sy_task = compute_covariance(np.reshape(eval_y, (-1, config['num_rnn_out'])).T)
                    if config['debug_timing']:
                        print("Sy_task", time.time()-t_start)

                    # get block matrices from Shx_task
                    # Sh_task = Shx_task[-hp['n_rnn']:, -hp['n_rnn']:]
                    Sh_task = np.matmul(np.matmul(Wfull.T, Shx_task), Wfull)
                    if config['debug_timing']:
                        print("Sh_task", time.time()-t_start)

                    # ---------- update stored covariance matrices for continual learning -------
                    taskNumber = len(convCnt)
                    if taskNumber == 0:
                        input_cov = Shx_task
                        activity_cov = Sh_task
                        output_cov = Sy_task

                        if config['debug_clmult']:
                            np.save('temp/eval_h.npy', eval_h)
                            np.save('temp/eval_x.npy', eval_x)
                            np.save('temp/eval_y.npy', eval_y)
                            np.save('temp/Win.npy', Win)
                            np.save('temp/Wrec.npy', Wrec)
                            np.save('temp/full_state.npy', full_state)
                            np.save('temp/Wfull.npy', Wfull)
                            np.save('temp/Shx_task.npy', Shx_task)
                            np.save('temp/Sy_task.npy', Sy_task)
                            np.save('temp/Sh_task.npy', Sh_task)
                    else:
                        input_cov = taskNumber / (taskNumber + 1) * input_cov + Shx_task / (taskNumber + 1)
                        activity_cov = taskNumber / (taskNumber + 1) * activity_cov + Sh_task / (taskNumber + 1)
                        output_cov = taskNumber / (taskNumber + 1) * output_cov + Sy_task / (taskNumber + 1)

                    # ---------- update projection matrices for continual learning ----------
                    activity_proj, input_proj, output_proj, recurrent_proj = compute_projection_matrices(activity_cov, input_cov, output_cov, input_cov[-config['num_rnn']:, -config['num_rnn']:], config["alpha_projection"])
                    if config['debug_timing']:
                        print("compute_proj_matrices", time.time()-t_start)

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
                            with open(os.path.join('data', config['save_dir'], 'perfs_' + str(len(convCnt)) + '.pkl'),'wb') as file:
                                pickle.dump(test_perfs, file)

                    # Save problem learning-specific stat summary
                    wNormR.append(np.mean(wNR[-50:]))
                    wNormR2.append(np.mean(wNR2[-50:]))
                    wNormI.append(np.mean(wNI[-50:]))
                    wNormO.append(np.mean(wNO[-50:]))
                    hNorm.append(np.mean(hN[-50:]))
                    HM.append(max(hm[-50:]))
                    np.savetxt(os.path.join('data', config['save_dir'], 'trIms_' + str(len(convCnt)) + '.txt'), np.array(trIm), fmt='%f', delimiter=' ')

                    # Set firing rate homeostatic set point after first problem is learned
                    if firstConv == False:
                        firstConv = True
                        model.updateRegularizerTargets(hNorm[-1], wNormR[-1], wNormI[-1], sess)

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
                        f.write(f'{convCnt[-1]:f}\n')
                    with open(os.path.join('data', config['save_dir'], 'wNormR.txt'), 'a') as f:
                        f.write(f'{wNormR[-1]:12.9f}\n')
                    with open(os.path.join('data', config['save_dir'], 'wNormR2.txt'), 'a') as f:
                        f.write(f'{wNormR2[-1]:12.9f}\n')
                    with open(os.path.join('data', config['save_dir'], 'wNormI.txt'), 'a') as f:
                        f.write(f'{wNormI[-1]:12.9f}\n')
                    with open(os.path.join('data', config['save_dir'], 'wNormO.txt'), 'a') as f:
                        f.write(f'{wNormO[-1]:12.9f}\n')
                    with open(os.path.join('data', config['save_dir'], 'hNorm.txt'), 'a') as f:
                        f.write(f'{hNorm[-1]:12.9f}\n')
                    with open(os.path.join('data', config['save_dir'], 'HM.txt'), 'a') as f:
                        f.write(f'{HM[-1]:12.9f}\n')
                    with open(os.path.join('data', config['save_dir'], 'SINGS.txt'), 'a') as f:
                        f.write('  '.join([f'{val:12.9f}' for val in singVals[-1]]) + '\n')
                
                if config['debug_timing']:
                    print("block1", time.time()-t_start)

                if config['runType'] != runType.Full:  # for manifold perturbation only
                    st0 = getStates(sess, config, model, images)
                    if len(convCnt) <= 50:
                        X = st0[0]
                        print('Size: ' + str(X.shape))
                        saveStates[:,:,len(convCnt)-1] = X.T
                        X = st0[1]
                        saveStates[:,:,len(convCnt)-1+50] = X.T
                        
                # Summarize and print learning stats for learned problem
                print('Converged in: ' + str(convCnt[-1]) + ' ' + str(len(convCnt)) + ' (' + str(
                    wNormR[-1]) + ' ' +  str(wNormR2[-1]) + ' ('+ str(maxSingVal) +'), ' + str(wNormI[-1]) + ', ' + str(wNormO[-1]) + ') (' + str(hNorm[-1]) + ', ' + str(HM[-1]) + ') ' + str(np.mean(perf[-50:]))+ ' ' + str(topTenSings) )
                print('Sing Dev: ' + str(np.sum(np.abs(singVals[0,:]-currSingVals[0]))))

                if config['runType'] != runType.Full:  # for manifold perturbation only
                    if len(convCnt) == 50:
                        np.save(os.path.join('data', config['save_dir'], 'states'), saveStates)

                        model.restore(1)
                        oldWts, dEigs, sEigs = resetOutWeightsWithSameStruct(sess, model, config, 4, config['runType'], states=saveStates)
                        model.removeTrainable(['out_RNN_weights', 'out_RNN_biases'], config)
                        model.printTrainable()

                    if len(convCnt) >= 50:
                        print(str(convCnt[-1]))
                        model.restore(1)
                        resetOutWeightsWithSameStruct(sess, model, config, 4, config['runType'], eigs=dEigs, eigsS=sEigs)
                        model.printTrainable()
                
                sys.stdout.flush()

                # The orthogonalization code runs if this is commented out (otherwise I get an error about the graph already being finalized.)
                # It seems like it's okay to comment out as long as the code runs on a single thread?
                # https://www.tensorflow.org/api_docs/python/tf/Graph#finalize
                # if config['runType'] == runType.Full:
                #     # Finalize graph after homeostatic set point is set
                #     if len(convCnt) == 1:
                #         sess.graph.finalize()

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

                if config['debug_timing']:
                    print("end", time.time()-t_start)

            # Done learning all problems?
            if len(convCnt) >= config['max_tasks']:
                break

    print(convCnt)