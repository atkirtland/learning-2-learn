import sys
sys.path.append(".")

from train import train
from train import runType

if __name__ == '__main__':
    train(seed          = 0,
          batchSize     = 1,
          l2            = 0.0005,
          l2_wR          = 0.001,
          l2_wI          = 0.0001,
          l2_wO          = 0.1,
          svBnd = 10.0,
          rType = runType.Full,

          projGrad = False,
          originalAdam = False,
          learningRateInit = 0.0001,
          beta1 = 0.3,
          beta2 = 0.999)