import numpy as np
import cv2
from tqdm import tqdm
import time
from scipy.special import expit
from scipy import optimize
import signal
import glob
import os
from Adam1 import initialize_adam, update_parameters_with_adam
# from scipy.misc import imread

def adam(params, vs, sqrs, lr, batch_size, t):
    beta1 = 0.9
    beta2 = 0.999
    eps_stable = 1e-8

    for param, v, sqr in zip(params, vs, sqrs):
        g = param.grad / batch_size

        v[:] = beta1 * v + (1. - beta1) * g
        sqr[:] = beta2 * sqr + (1. - beta2) * nd.square(g)

        v_bias_corr = v / (1. - beta1 ** t)
        sqr_bias_corr = sqr / (1. - beta2 ** t)

        div = lr * v_bias_corr / (nd.sqrt(sqr_bias_corr) + eps_stable)
        param[:] = param - div

def train(path_to_images, csv_file):
    data = np.genfromtxt(csv_file, delimiter = ',')
    # data = np.fromstring(csv_file, sep = ',')
    frame_nums = data[:,0]
    steering_angles = data[:,1]
    theta = np.matrix(steering_angles).transpose()
    max_angle = 180
    min_angle = -180
    bins = np.linspace(min_angle,max_angle,64) 
    values = np.linspace(0,64 -1,64)
    y = np.zeros((len(theta),64))
    for i,angle in enumerate(theta):
        index =  int(np.interp(angle,bins, values))
        y[i,index] = 1
        
        # Creating a Gaussian distribution with a center on the target index 
        # Example of Gaussian Distribution [0 0.3 0.6 0.89 1 0.89 0.6 0.3 0]
        if index - 1 >= 0 and index + 1 < 64:
            y[i,index - 1] = 0.89
            y[i,index + 1] = 0.89
            
            if index - 2 >= 0 and  index + 2 < 64:
                y[i,index - 2] = 0.6
                y[i,index + 2] = 0.6
                
                if index - 3 >= 0 and  index + 3 < 64:
                    y[i,index - 2] = 0.3
                    y[i,index + 2] = 0.3
    X=[]             
    dim = (60, 60)   
    for i in range(frame_nums.shape[0]):
        im_full = cv2.imread(path_to_images + '/' + str(int(i)).zfill(4) + '.jpg')
        im_full = im_full[:, :, 2]
        im_full = cv2.resize(im_full, dim)     
        X.append(np.array(im_full[30:,:])/255)  
            
    # You could import your images one at a time or all at once first, 
    # here's some code to import a single image:
    iterations = 3300
    learning_rate = 1e-1*(4)
    NN = NeuralNetwork() 
    X = np.reshape(X,(1500,(1800)))

    T = trainer(NN)
    T.train(X,y)
    return NN


def predict(NN, image_file):
    '''
    Second method you need to complete. 
    Given an image filename, load image, make and return predicted steering angle in degrees. 
    '''
    im_full = cv2.imread(image_file)
    bins = np.linspace(min_angle,max_angle,64) 
    im_full = cv2.imread(image_file)
    im_full = im_full[:, :, 2]
    # Resizing image
    im_full = cv2.resize(im_full, (60,60))
    # Croping image
    image_vector = np.array(im_full[30:,:])/255
    # Flatting image
    image_vector = np.reshape(image_vector,(1,-1))
    # Return normalised image
    return bins[np.argmax(NN.forward(image_vector))]

class NeuralNetwork(object):
    def __init__(self, Lambda=0):        
        '''
        Neural Network Class, you may need to make some modifications here!
        '''
        self.inputLayerSize = 1800
        self.outputLayerSize = 64
        self.hiddenLayerSize = 128
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
        self.Lambda = Lambda
    
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda*self.W2
        
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.W1
         
        
        return dJdW1, dJdW2
    
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
    
    
    
    
class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))
        self.testJ.append(self.N.costFunction(self.testX, self.testY))
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        
        return cost, grad
        
#     def train(self, trainX, trainY, testX, testY):
    def train(self, X, y):
        #Make an internal variable for the callback function:
#         self.X = trainX
#         self.y = trainY
        
#         self.testX = testX
#         self.testY = testY
        self.X = X
        self.y = y

        #Make empty list to store training costs:
        self.J = []
#         self.testJ = []
        
        params0 = self.N.getParams()
        print(params0)

        options = {'maxiter': 200, 'disp' : True}
#         _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS',args=(trainX, trainY), options=options, callback=self.callbackF)
        params0 = self.N.getParams()
        print(params0.shape)
        e0 = params0.reshape(-1, 1)
        # a = adam()
        # a =Adam1()
        m, v = initialize_adam(params0,3) 
        _res = update_parameters_with_adam(params0,3,grad, m, v)
        
        # _res = optimize.minimize(self.costFunctionWrapper, e0, jac=True, method='BFGS',args=(X, y), callback=self.callbackF, options=options)

        self.N.setParams(_res.parameters)
        self.optimizationResults = _res   
    
    