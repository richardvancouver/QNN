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




class layer():#creating an obj, defined by three attributes/items/contents: n nodes in the entry 'layer'; m nodes in the exit 'layer'; matrix connecting n to m

    def __init__(self, n, m, comatrix=np.array([]), delta=0, shiftlayer=False):
        self.nin=n
        self.nout=m

        self.shiftlayer=shiftlayer
        if len(comatrix):
            self.theta = comatrix
        else:

            if not delta:
                delta=np.sqrt(6)/np.sqrt(n+m)

            self.randomize(delta)

        #self.shiftlayer=shiftlayer #need to define it earlier than self.randomize, otherwise when going through self.randomize(delta), error will occur
        print(self.shiftlayer)

    def randomize(self, delta=None):
        N=self.nin
        M=self.nout

        if not delta:
            delta = np.sqrt(6)/np.sqrt(N+M)

        self.theta = np.random.uniform(-delta, delta, N*M).reshape(N,M) #fill the matrix with random numbers


        print(self.shiftlayer)
        if self.shiftlayer:  #another way of filling the matrix, make the filled elements follow some format
            l1=self.theta[:,0]
            dshift=N//M
            for i in range(M-1):
                shift = dshift*(i+1)
                self.theta[:,i+1]=np.roll(l1,shift)





class neuralnewtork(BaseEstimator):


    def __init__(self, thetas=None, design=None, shiftlayer=None):

        
        self.design=design #design gives the layout illustration
        self.shiftlayer=shiftlayer
        if thetas!= None:

            nfeatures = thetas[0].shape[0]-1  #extract the number of features from the first theta, theta[0]
            ntargets = thetas[-1].shape[1]    #extract the number of targets from the last theta, theta[-1]
            self.ntargets=ntargets            #features-------------->targets, the stuff hidden in between are the hidden layers
            self.createlayers(nfeatures,ntargets, thetas=thetas, shiftlayer=shiftlayer)



    def createlayers(self, nfeatures, ntargets, thetas=None, design=None,shiftlayer=None):

        if design==None:
            if self.design!=None:
                design=self.design
            else:
                design=[16]

        if isinstance(design, type(int())):
            design=[design]

        if thetas!=None:
            design=[]
            for theta in thetas[:-1]:
                design.append(theta.shape[1])
        
        self.design=design





        layers=[]
        nl=len(design)+1 #design contains exports, doesn't include raw features in, design only contains hidden layers


        for idx in range(nl):#add layer objs to layers, iterate through the layers, each layer has its own connecting matrix theta, lin and lout

            #fetch theta for layer idx
            if thetas!=None:
                theta=thetas[idx]
            else:
                theta=np.array([])

            #set up biases:
            if idx==0:
                lin = nfeatures + 1
            else:
                lin = design[idx-1]+1

            if idx!=nl-1:
                lout=design[idx]
            else:
                lout=ntargets


           #now, fill the layers with layer objects:     


            if shiftlayer!= None:
                if idx == shiftlayer:
                    layers.append(layer(lin, lout, theta, shiftlayer=True))
                else:
                    layers.append(layer(lin, lout, theta, shiftlayer=False))

            else: layers.append(layer(lin, lout, theta, shiftlayer=False))

        self.layers=layers

        self.nlayers=len(layers)




    def propogatefwd(self, z, nl=100): #propogate through the network to get the response
        #pass

        if isinstance(z, type([])): #type check
            z=np.array(z)


        #add bias
        if z.ndim==2:
            N=z.shape[0]

            a=np.hstack([np.ones(N,1),z])

        else:
            N=1
            a=np.hstack([np.ones(N),z] )






        final_layer = len(self.layers)-1
        for lidx, lv in enumerate(self.layers[0:nl]):
            z = np.dot(a, lv.theta)

            if lidx != final_layer:

                if N == 1 and z.ndim == 1:
                    a = np.hstack([np.ones(N), sigmoid(z)])
                else:
                    a = np.hstack([np.ones((N, 1)), sigmoid(z)])

            else:

                a=sigmoid(z)


        
        return z, a



def sigmoid(z):
    """
    compute element-wise the sigmoid of input array

    """
    return 1./(1.0 + np.exp(-z))






if __name__ =='__main__':#test unit
    #test class 1
    data = handwritingpixsk(samples='ex4data1.mat')
    ################
    # show some samples
    #data.plot_samples(15)

    #test class 2
    data2 = handwritingpix(samples='ex4data1.mat',thetas='ex4weights.mat')
    ################
    # show some samples
    #data2.plot_samples(15)

    #test the layer class
    test1=layer(5, 3, comatrix=np.array([]), delta=0, shiftlayer=True)
    #print("%s %s",%(test1.nin, test1.nout))
    print(test1.nin)
    print(test1.nout)
    print(test1.theta)

    nfeaturesp1=5+1
    delta=0.1
    ntargets=3
    testnn1=neuralnewtork(thetas=[ np.random.uniform(-delta, delta, nfeaturesp1*5).reshape(nfeaturesp1,5), np.random.uniform(-delta, delta, (5+1)*3).reshape(5+1,3), np.random.uniform(-delta, delta, (3+1)*2).reshape(3+1,2), np.random.uniform(-delta, delta, (2+1)*ntargets).reshape(2+1,ntargets) ])

    print(testnn1.design)