'''
Created on Jan 20, 2018

@author: hshi
'''
from Coordinate_3D import Coordinate_3D
from Coordinate import Coordinate
from Orientation import Orientation
class Joint(object):
    '''
    classdocs
    '''


    def __init__(self, jointInfo=None, X_3D=None, Y_3D=None, Z_3D=None, X_depth=None, Y_depth=None, X_rgb=None, Y_rgb=None, 
                 W_orientation=None, X_orientation=None, Y_orientation=None, Z_orientation=None, trackingState = None):
        '''
        Constructor
        '''
        if jointInfo is not None:
            self.coordinate_3D = Coordinate_3D(jointInfo[0], jointInfo[1], jointInfo[2])
            self.coordinate_depth = Coordinate(jointInfo[3], jointInfo[4])
            self.coordinate_rgb = Coordinate(jointInfo[5], jointInfo[6])
            
            self.orientation = Orientation(jointInfo[7], jointInfo[8], jointInfo[9], jointInfo[10])
            self.trackingState = jointInfo[11]
        else:
            self.coordinate_3D = Coordinate_3D(X_3D, Y_3D, Z_3D)
            self.coordinate_depth = Coordinate(X_depth, Y_depth)
            self.coordinate_rgb = Coordinate(X_rgb, Y_rgb)
            
            self.orientation = Orientation(W_orientation, X_orientation, Y_orientation, Z_orientation)
            self.trackingState = trackingState
            
            
            
    def getCoordinate_3D(self):
        return self.coordinate_3D.getCoordinate()
    
    def getCoordinate_depth(self):
        return self.coordinate_depth.getCoordinate()
    
    def getCoordinate_rgb(self):
        return self.coordinate_rgb.getCoordinate()
    
    def getOrientation(self):
        return self.orientation.getOrientation()
    
    def getTrackingState(self):
        return self.trackingState

    