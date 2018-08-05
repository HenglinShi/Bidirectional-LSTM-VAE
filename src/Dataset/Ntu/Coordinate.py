'''
Created on Jan 20, 2018

@author: hshi
'''

class Coordinate(object):
    '''
    classdocs
    '''


    def __init__(self, X, Y):
        '''
        Constructor
        '''
        self.X = X
        self.Y = Y
        
    def getX(self):
        return self.X
    
    def getY(self):
        return self.Y
    
    def getCoordinate(self):
        return self.getX(), self.getY()