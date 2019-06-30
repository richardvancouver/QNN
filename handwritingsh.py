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


        if nl>self.nlayers:
            print("error: nl must be smaller than number of layers")
            raise ValueError("error: nl must be smaller than number of layers")

        else:
            final_layer = len(self.layers)-1
            for lidx, lv in enumerate(self.layers[0:nl]):
                z = np.dot(a, lv.theta)

                if lidx != final_layer:#when not the final layer, needs to add the bias 

                    if N == 1 and z.ndim == 1:
                        a = np.hstack([np.ones(N), sigmoid(z)])
                    else:
                        a = np.hstack([np.ones((N, 1)), sigmoid(z)])

                else:

                    a=sigmoid(z)


        
        return z, a #return the raw output z, as well as the sigmoided output a=sigmoid(z)



    def costfunction(self,flatthetas,x,y, nll=100):#need to make flatthetas show up, if you want to use the built-in optimizer such as scipy.minimize


        if isinstance(x,type([])):
            X=np.array(x)

        #
        self.unflatten_thetas(flatthetas) #need this if the input thetas is flattened, in the fit procedure, need to feed the flattened thetas to grad 
                                        #also a way to reset the self.thetas during the optimization iterations
        if X.ndim == 2:
            N = X.shape[0] 
        else:
            N = 1


        z, h = self.propogatefwd(X, nll)
        yy = labels2vectors(y, self.ntargets)

        J = 0.
        J = (-np.log(h) * yy.transpose() - np.log(1-h)*(1-yy.transpose())).sum()
        J = J/N


        return J



    def unflatten_thetas(self, flatthetas):#take flatthetas, set the unflattened values to self.thetas
        """
        in order to use scipy.fmin functions we 
        need to make the Theta dependance explicit.
        
        This routine takes a flattened array 'thetas' of
        all the internal layer 'thetas', then assigns them
        onto their respective layers
        (ordered from earliest to latest layer)

        """
        bi = 0
        for lv in self.layers:
            shape = lv.theta.shape
            ei = bi + shape[0] * shape[1]
            lv.theta = flatthetas[bi:ei].reshape(shape)
            bi = ei

    def flatten_thetas(self, thetas=None):
        """
        in order to use scipy.fmin functions we
        need to make the layer's Theta dependencies explicit.

        this routine returns a giant array of all the flattened
        internal theta's, from earliest to latest layers

        """
        z = np.array([])
        if not thetas:
            for lv in self.layers:
                z = np.hstack([z, lv.theta.flatten()])#originally only with this, thus flatten_thetas is kinda not meant to be used as a standalone func, i.e. need to re-instantiate the instance to reset thetas
        else:
            for tta in thetas:
                z = np.hstack([z,tta.flatten()]) #now, give it a choice to run as an standalone func, pass thetas in and returns the flattened thetas


        return z



    def gradient(self,flatthetas,x,y, nll=None):#flatthetas

        if isinstance(x, type([])):
            X = np.array(x)
        print("X:",X)
        N = X.ndim
        print("N:",N) #should be 1 or 2?
        nl = len(self.layers)#nl = self.nlayers

# create our grad_theta arrays (init to zero):
        grads = {}#np.array([])#{}
        for li, lv in enumerate(self.layers):
            grads[li] = np.zeros_like(lv.theta)
            print("grad[li]:",grads[li])

        #if 1:
        for li in range(nl, 0, -1): #grad(a_previous, delta_current)
            z, a = self.propogatefwd(X,li)

            if li == nl:
                ay = labels2vectors(y, self.ntargets).transpose()
                delta = (a - ay)
                print("delta at li equals nl:",delta)
            else:
                theta = self.layers[li].theta
                print("theta in tmp branch:",theta)
                print("sigmoidGrad:",sigmoidGradient(z))
                print("np.ones(N,1):",np.ones(N) )
                aprime = np.hstack([np.ones(N), sigmoidGradient(z)]) #add in bias  np.ones((N,1))
#use fortran matmult if arrays are large
                # if _fort_opt and deltan.size * theta.size > 100000:
                #     tmp = matmult(deltan,theta.transpose())
                # else:
                #     tmp = np.dot(deltan,theta.transpose())#nsamples x neurons(li)

                tmp = np.dot(deltan,theta.transpose()) #matrix
                print("tmp:", tmp)
                print("aprime:", aprime)
                delta = tmp*aprime #aprime is a number? no, a col-vec
                print("delta:",delta)
                
#find contribution to grad
            idx = li - 1
            z, a = self.propogatefwd(X,idx)
            print("a.transpose() before times by delta:",a.transpose())
            if idx in grads:
                if li == nl:
                    grads[idx] = np.outer(a.transpose(), delta)/N #to get a matrix grads[idx], we should do np.outer, since a is a col-vec, delta should also be a col-vec
                else:
                    #strip off bias
                    grads[idx] = np.outer(a.transpose(), delta[1:])/N

#if this is a "shift-invar" layer, find the average grad 
            # if self.shiftlayer is not None:
            #     if li == self.shiftlayer:
            #         shape = self.layers[li].theta.shape
            #         grads = grads.reshape(shape)
            #         l1 = grads[:,0]
            #         dshift = N//m
            #         for i in range(m-1):
            #             shift = -dshift * (i+1) #undo the previous shifts
            #             self.theta[:,i+1] = np.roll(l1, shift)
            #         grad_avg = self.theta.mean(axis=0)
            #         grads = np.array([grad_avg for i in range(shape[1])]).flatten()

         

#keep this delta for the next (earlier) layer
            if li == nl:
                deltan = delta
            else:
                deltan = delta[1:]

#finally, flatten the gradients
        z = np.array([])
        for k in sorted(grads):
            v = grads[k]
            z = np.hstack([z, v.flatten()]) #each grads[k] should be a connecting matrix
        return z


    def numericalGradients(self, X, y):
        """
        numerically estimate the gradients using finite differences
        (used to compare to 'gradient' routine)

        loop over layers, perturbing each theta-parameter one at a time


        * useful for testing gradient routine *
        """
        from copy import deepcopy

        thetas = self.flatten_thetas()
        origthetas = deepcopy(thetas)
        numgrad = np.zeros(thetas.size)
        perturb = np.zeros(thetas.size)

        delta = 1.e-4
        nl=self.nlayers
        for p in range(numgrad.size):
            #set the perturbation vector
            perturb[p] = delta
            loss1 = self.costfunction(thetas - perturb, X, y, nll=nl)
            loss2 = self.costfunction(thetas + perturb, X, y, nll=nl)
            #calculat the numerical gradient
            numgrad[p] = (loss2 - loss1) / (2*delta)
            #reset the perturbation
            self.unflatten_thetas(origthetas)
            perturb[p] = 0
            
## OLD
# loop over layers, neurons
        idx = 0
#         if 0: 
#             for lv in self.layers:
#                 theta_orig = deepcopy(lv.theta)

# # perturb each neuron and calc. grad at that neuron
#                 for pi in range(theta_orig.size): #strip bias
#                     perturb = np.zeros(theta_orig.size)
#                     perturb[pi] = delta
#                     perturb = perturb.reshape(theta_orig.shape)
#                     lv.theta = theta_orig + perturb
#                     loss1 = self.costFunctionU(X, y)
#                     lv.theta = theta_orig - perturb
#                     loss2 = self.costFunctionU(X, y)
#                     numgrad[idx] = (loss2 - loss1) / (2*delta)
#                     idx += 1
#                 lv.theta = theta_orig

        return numgrad





def sigmoid(z):
    """
    compute element-wise the sigmoid of input array

    """
    return 1./(1.0 + np.exp(-z))



def sigmoidGradient(z):
    """
    compute element-wise the sigmoid-Gradient of input array

    """
    return sigmoid(z) * (1-sigmoid(z))
        



def labels2vectors(y, Nclass=1):
    """
    given a vector of [nsamples] where the i'th entry is label/classification
    for the i'th sample, return an array [nlabels,nsamples],
    projecting each sample 'y' into the appropriate row

    args:
    y : [nsamples] 
    Nclass: the number of classifications,
            defaults to number of unique items in y

    **assumes classifications are in range 0 <= y < Nclass
    """
    if isinstance(y, np.array([]).__class__):
        pass
    else:
        y = np.array([y], dtype=np.int)

# number of samples
    N = y.size

# determine number of classes
    if Nclass:
        nclass = Nclass
    else:
        nclass = len(np.unique(y))


# map labels onto column vectors
    if N == 1:
        yy = np.zeros(nclass, dtype=np.uint8)
    else:
        yy = np.zeros((nclass, N), dtype=np.uint8)

    for yi, yv in enumerate(y):
        if N == 1:
            yy[yv] = 1
        else:
            yy[yv, yi] = 1
    return yy



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
    print("number of layers:",testnn1.nlayers)
    response1=testnn1.propogatefwd([2,2,2,2,2], nl=4)
    print("propogate through 4 layers of the nn:", response1)

    flatthetas=testnn1.flatten_thetas()
    print("flattened thetas:",flatthetas)
    cost=testnn1.costfunction(flatthetas,[2,2,2,2,2],[2],4)
    print("cost:", cost)

    #teststandaloneflat=testnn1.flatten_thetas( [ np.random.uniform(-delta, delta, nfeaturesp1*5).reshape(nfeaturesp1,5), np.random.uniform(-delta, delta, (5+1)*3).reshape(5+1,3), np.random.uniform(-delta, delta, (3+1)*2).reshape(3+1,2), np.random.uniform(-delta, delta, (2+1)*ntargets).reshape(2+1,ntargets) ] )
    teststandaloneflat=testnn1.flatten_thetas( [ np.random.uniform(-delta, delta, nfeaturesp1*5).reshape(nfeaturesp1,5), np.random.uniform(-delta, delta, (5+1)*3).reshape(5+1,3), np.random.uniform(-delta, delta, (3+1)*2).reshape(3+1,2)] )

    print("test flatten_thetas:", teststandaloneflat)


    testgrad=testnn1.numericalGradients([2,2,2,2,2],[2])
    print("testgrad:",testgrad)

    test_non_num_grad=testnn1.gradient([],[2,2,2,2,2],[2])
    print("test non-numerical gradient:", test_non_num_grad)



    #for fimin_cg to work, costfunction and gradient should have the same input arguments, therefore, modify the arguments list a little bit, added some dummy arguments
    thetas1 = testnn1.flatten_thetas()
    xopt = fmin_cg(f=testnn1.costfunction,x0=thetas1,fprime=testnn1.gradient,args=([2,2,2,2,2],[2],4), maxiter=100,epsilon=0.1,gtol=0.1,disp=0)

    print("xopt:",xopt)