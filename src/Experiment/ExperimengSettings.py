'''_
Created on Apr 18, 2018

@author: hshi
'''
import os
workingDirName = '20180419_4'
rootWorkingDir = '/wrk/hshi/DONOTREMOVE/git/FeatureLearningAndGestureRecognition/ExperimentArchive/New'
workingDir = os.path.join(rootWorkingDir, workingDirName)

if not os.path.exists(workingDir):
    os.mkdir(workingDir)
    
    

