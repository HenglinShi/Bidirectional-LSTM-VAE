'''
Created on Apr 22, 2018

@author: hshi
'''
import scipy.io as sio
import numpy as np
import os


dir_ = './train'

fileList = os.listdir(dir_)
totalL = 0
for i in range(len(fileList)):
    curP = os.path.join(dir_, fileList[i])
    newMat = sio.loadmat(curP)['sample']
    totalL = totalL + newMat.shape[0]
    print i
    

dir_ = './test'

fileList = os.listdir(dir_)

for i in range(len(fileList)):
    curP = os.path.join(dir_, fileList[i])
    newMat = sio.loadmat(curP)['sample']
    totalL = totalL + newMat.shape[0]
    print i





















mMat = np.zeros([totalL, 2775])
end = 0




dir_ = './train'

fileList = os.listdir(dir_)

for i in range(len(fileList)):
    curP = os.path.join(dir_, fileList[i])
    newMat = sio.loadmat(curP)['sample']
    beg = end
    end = end + newMat.shape[0]
    
    mMat[beg:end, :] = newMat
    print i
    

dir_ = './test'

fileList = os.listdir(dir_)

for i in range(len(fileList)):
    curP = os.path.join(dir_, fileList[i])
    newMat = sio.loadmat(curP)['sample']
    
    beg = end
    end = end + newMat.shape[0]
    
    mMat[beg:end, :] = newMat
    print i
from sklearn import preprocessing

mMean = newMat.mean(0)
for i in range(mMat.shape[0]):
    mMat[i] = mMat[i] - mMean
    

mStd = newMat.std(0)
import scipy.io as sio

sio.savemat('mPrior.mat', {'mean': mMean, 'std': mStd})
priorScalar = preprocessing.StandardScaler().fit(mMat)

import cPickle as pickle
priorFile = 'prior.pkl'
f = open(priorFile, 'wb')
pickle.dump(priorScalar, f)
f.close()


