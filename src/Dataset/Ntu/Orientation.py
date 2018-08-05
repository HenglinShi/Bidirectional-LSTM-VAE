'''
Created on Jan 20, 2018

@author: hshi
'''

class Orientation(object):
    '''
    classdocs
    '''


    def __init__(self, W, X, Y, Z):
        '''
        Constructor
        '''
        self.W = W
        self.X = X
        self.Y = Y
        self.Z = Z
        
    def getW(self):
        return self.W
    
    def getX(self):
        return self.X
    
    def getY(self):
        return self.Y
    
    def getZ(self):
        return self.Z
    
    def getOrientation(self):
        return self.getW(), self.getX(), self.getY(), self.getZ()