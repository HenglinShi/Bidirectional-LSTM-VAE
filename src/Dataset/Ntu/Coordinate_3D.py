'''
Created on Jan 20, 2018

@author: hshi
'''
from Coordinate import Coordinate
class Coordinate_3D(Coordinate):
    '''
    classdocs
    '''


    def __init__(self, X, Y, Z):
        '''
        Constructor
        '''
        Coordinate.__init__(self, X, Y)
        self.Z = Z
        
        
    def getZ(self):
        return self.Z
    
    def getCoordinate(self):
        return self.getX(), self.getY(), self.getZ()