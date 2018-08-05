'''
Created on Jan 20, 2018

@author: hshi
'''

class Frame(object):
    '''
    classdocs
    '''


    def __init__(self, frameID):
        '''
        Constructor
        '''
        self.frameID = frameID
        self.skeletonNum = 0
        
        
        self.skeletonList = {}
        
        
    def appendSkeleton(self, skeleton):
        
        self.skeletonList[skeleton.bodyId] = skeleton
        self.skeletonNum += 1
        
        
    def getCoordinates_3D(self, bodyId, selectedJoint = None):
        if bodyId not in self.skeletonList.keys():
            return None
        else:
            return self.skeletonList[bodyId].getCoordinates_3D(selectedJoint)
        
        
    def getSkeleton(self, bodyId):
        return self.skeletonList[bodyId]
    
    def getBodyIds(self):
        return self.skeletonList.keys()
    
    def getBodyNum(self):
        return len(self.skeletonList.keys())
    
    
    def getData(self, bodyId, dataType = '3dCoordinate rotation 2dCoordinate depthCoordinate', selectedJoint=None):
        if bodyId not in self.skeletonList.keys():
            return None
        else:

            return self.skeletonList[bodyId].getData(selectedJoint, dataType)
            
            
    