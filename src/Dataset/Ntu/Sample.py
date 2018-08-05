'''
Created on Jan 20, 2018

@author: hshi
'''
import os
from Joint import Joint
import numpy as np
from Skeleton import Skeleton
from Frame import Frame

class Sample(object):
    '''
    classdocs
    '''


    def __init__(self, sampleFilePath):
        '''
        Constructor
        '''
        self.sampleFilePath = sampleFilePath
        self.frameNum = 0
        self.frameList = list()
        self.parsing()
        
    def parsing(self):
        
        with open(self.sampleFilePath, 'r') as f:
            
            frameCount = f.readline().split(' ')
            frameCount = int(frameCount[0])
                
                
                
                
            for frameIte in range(frameCount):
                    
                    
                currentFrame = Frame(frameIte)
                bodyCount = f.readline().split(' ')
                bodyCount = int(bodyCount[0])
                
                for _ in range(bodyCount):
                        
                    metaData = f.readline().strip().split(' ')
                    jointCount = f.readline().strip().split(' ')
                    jointCount = int(jointCount[0])
                        
                    currentSkeleton = Skeleton(jointCount, metaData)
                        
                    for jointIte in range(jointCount):
                        jointInfoLine = f.readline()
                        jointInfo = np.array(jointInfoLine.strip().split(' '), dtype = 'float')
                            
                        currentSkeleton.appendJoint(jointIte, jointVec = jointInfo)
                            #jointList.append(Joint(jointInfo))
                    
                        
                    currentFrame.appendSkeleton(currentSkeleton)
                        
                        
                self.frameList.append(currentFrame)
                self.frameNum += 1
                    
                    
                    