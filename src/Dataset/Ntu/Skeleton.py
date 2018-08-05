'''
Created on Jan 20, 2018

@author: hshi
'''
from Joint import Joint
import numpy as np
class Skeleton(object):
    '''
    classdocs
    '''


    def __init__(self, jointNum, metaData = None, bodyId = None, clipedEdges = None, 
                 handConfidence_left = None, handState_left = None, 
                 handConfidence_right = None, handState_right = None, 
                 isRestricted = None, 
                 leanX = None, leanY = None, trackingState = None, jointList = None):
      
        self.jointNum = jointNum
        self.jointList = list()
        
        for _ in range(self.jointNum):
            self.jointList.append(None)
      
        if metaData is not None:
            self.bodyId = int(metaData[0])
            self.clipedEdges = int(metaData[1])
            self.handConfidence_left = int(metaData[2])
            self.handState_left = int(metaData[3])
            self.handConfidence_right = int(metaData[4])
            self.handState_right = int(metaData[5])
            self.isRestricted = int(metaData[6])     
            self.leanX = float(metaData[7])
            self.leanY = float(metaData[8])  
            self.trackingState = int(metaData[9]) 
            
        else: 
            self.bodyId = bodyId
            self.clipedEdges = clipedEdges
            self.handConfidence_left = handConfidence_left
            self.handState_left = handState_left
            self.handConfidence_right = handConfidence_right
            self.handState_right = handState_right
            self.isRestricted = isRestricted
            self.leanX = leanX
            self.leanY = leanY
            self.trackingState = trackingState
            
    def updateJoint(self, jointInd, joint):
        self.jointList[jointInd] = joint
        
    
    def appendJoint(self, jointInd, joint = None, jointVec = None):
        if joint is not None:
            self.updateJoint(jointInd, joint)
            
        if jointVec is not None:
            self.updateJoint(jointInd, Joint(jointVec))
    
    def getCoordinates_3D(self, selectedJoints = None):
        if selectedJoints is None:
            selectedJoints = np.linspace(0, self.jointNum - 1, self.jointNum, dtype = 'int')
        selectedJointNum = len(selectedJoints)
        
        coordinates = np.zeros([3 * selectedJointNum])
        
        for jointIte in range(selectedJointNum):
            coordinates[jointIte * 3 : (jointIte + 1) * 3] = self.jointList[selectedJoints[jointIte]].getCoordinate_3D()
        
        return coordinates
        
    def getData(self, selectedJoints = None, dataType = '3dCoordinate rotation 2dCoordinate depthCoordinate'):
        if selectedJoints is None:
            selectedJoints = np.linspace(0, self.jointNum - 1, self.jointNum, dtype = 'int')
        selectedJointNum = len(selectedJoints)
        
        
        if dataType == '3dCoordinate rotation 2dCoordinate depthCoordinate':
            coordinates = np.zeros([11 * selectedJointNum])
            for jointIte in range(selectedJointNum):
                coordinates[jointIte * 11 : jointIte * 11 + 3] = self.jointList[selectedJoints[jointIte]].getCoordinate_3D()
                coordinates[jointIte * 11 + 3 : jointIte * 11 + 7] = self.jointList[selectedJoints[jointIte]].getOrientation()
                coordinates[jointIte * 11 + 7 : jointIte * 11 + 9] = self.jointList[selectedJoints[jointIte]].getCoordinate_rgb()
                coordinates[jointIte * 11 + 9 : jointIte * 11 + 11] = self.jointList[selectedJoints[jointIte]].getCoordinate_depth()
          
        
        if dataType == 'all':
            coordinates = np.zeros([12 * selectedJointNum])
            for jointIte in range(selectedJointNum):
                coordinates[jointIte * 12 : jointIte * 12 + 3] = self.jointList[selectedJoints[jointIte]].getCoordinate_3D()
                coordinates[jointIte * 12 + 3 : jointIte * 12 + 7] = self.jointList[selectedJoints[jointIte]].getOrientation()
                coordinates[jointIte * 12 + 7 : jointIte * 12 + 9] = self.jointList[selectedJoints[jointIte]].getCoordinate_rgb()
                coordinates[jointIte * 12 + 9 : jointIte * 12 + 11] = self.jointList[selectedJoints[jointIte]].getCoordinate_depth()
                coordinates[jointIte * 12 + 11] = self.jointList[selectedJoints[jointIte]].getTrackingState()
            
        return coordinates
    
    