import numpy as np
import _pickle as cPickle
#import cPickle
import pylab as plt
from scipy import io
from scipy import mgrid
from scipy.optimize import fmin_cg
import sys
from sklearn.base import BaseEstimator
from sklearn.datasets import load_digits


class handwritingpix(object):
    """
    read in the machine learning data files, which we 
    use to test our python implementation of the neural network

    we assume they are 20x20 pixels 
    initializes with a filename, assumes matlab format
    though can have text=True for text format
    then requires the trained values 'y'
    """
    def __init__(self, samples, thetas):

        self.fname = samples

        data = io.loadmat(samples)
        self.data = data['X']
        self.y = data['y']
# make sure the labels are in range 0<=y<nclass 
# so we can easily index our row vector
        self.y = self.y - self.y.min()
        thetas = io.loadmat(thetas)
        self.theta1 = thetas['Theta1'].transpose()
        self.theta2 = thetas['Theta2'].transpose()

#assume square image
        self.N = self.data.shape[1]
        self.Nsamples = self.data.shape[0]
        self.nx = np.sqrt(self.N)
        self.ny = np.sqrt(self.N)
        self.thetas = [self.theta1, self.theta2]

    def plot_samples(self, Nsamples=10):
        """
        randomly pick Nsamples**2 and plot them

        """
        nx = self.nx
        ny = self.ny

        n = np.random.uniform(0, self.Nsamples, Nsamples**2).astype(np.int)
        samples = self.data[n, :]
        
        data = np.zeros((Nsamples*int(ny), Nsamples*int(nx)))
        print(n)
        for xi, xv in enumerate(samples):
            col = xi % Nsamples
            row = xi // Nsamples
#            print xi,data.shape,row,col,xv.shape
            data[row*int(ny):(row+1)*int(ny), col*int(nx):(col+1)*int(nx)] = xv.reshape(20, 20)
            
        plt.imshow(data, cmap=plt.cm.gray)
        plt.show()
        print(xv.reshape(20,20))




class handwritingpixsk(object):
    """
    reads the handwritings dataset from sklearn
    
    
    """
    def __init__(self, samples):

        self.fname = samples

        data = load_digits()#io.loadmat(samples)
        self.data=data.images
        self.data2 = data.data #data['X']
        #self.y = data['y']
# make sure the labels are in range 0<=y<nclass 
# so we can easily index our row vector
        #self.y = self.y - self.y.min()
        #thetas = io.loadmat(thetas)
        #self.theta1 = thetas['Theta1'].transpose()
        #self.theta2 = thetas['Theta2'].transpose()

#assume square image
        self.N = self.data2.shape[1] #64
        self.Nsamples = self.data2.shape[0] #1797
        self.nx = np.sqrt(self.N)
        self.ny = np.sqrt(self.N)
        #self.thetas = [self.theta1, self.theta2]

    def plot_samples(self, Nsamples=10):
        """
        randomly pick up Nsamples**2 and plot them

        """
        nx = self.nx
        ny = self.ny

        n = np.random.uniform(0, self.Nsamples, Nsamples**2).astype(np.int)
        samples = self.data[n, :]
        
        data = np.zeros((Nsamples*int(ny), Nsamples*int(nx)))
        print(n)
        for xi, xv in enumerate(samples):
            col = xi % Nsamples
            row = xi // Nsamples
#            print xi,data.shape,row,col,xv.shape
            data[row*int(ny):(row+1)*int(ny), col*int(nx):(col+1)*int(nx)] = xv
            
        plt.imshow(data, cmap=plt.cm.gray)
        plt.show()
        print(xv)







if __name__ =='__main__':#test unit
    #test class 1
    data = handwritingpixsk(samples='ex4data1.mat')
    ################
    # show some samples
    data.plot_samples(15)

    #test class 2
    data2 = handwritingpix(samples='ex4data1.mat',thetas='ex4weights.mat')
    ################
    # show some samples
    data2.plot_samples(15)