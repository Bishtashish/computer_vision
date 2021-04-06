## ---------------------------- ##
##
## Example student submission code for autonomous driving challenge.
## You must modify the train and predict methods and the NeuralNetwork class. 
## 
## ---------------------------- ##

import numpy as np
import cv2
from tqdm import tqdm
import time
from scipy import optimize
# from scipy.misc import imread



Kx = np.array([[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]])
Ky = np.array([[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]])

def filter_2d(im, kernel):

    H, W = im.shape
    filtered_image = np.zeros((H-M+1, W-N+1), dtype='float64')
    for i in range(filtered_image.shape[0]):
        for j in range(filtered_image.shape[1]):
            filtered_image[i, j] = np.sum(
                np.multiply(im[i:i+M, j:j+N], kernel))

    return filtered_image

def sobel(gray):
    return [filter_2d(gray, Kx), filter_2d(gray, Ky)]

def train(path_to_images, csv_file):
    '''
    First method you need to complete. 
    Args: 
    path_to_images = path to jpg image files
    csv_file = path and filename to csv file containing frame numbers and steering angles. 
    Returns: 
    NN = Trained Neural Network object 
    '''

    # You may make changes here if you wish. 
    # Import Steering Angles CSV
    data = np.genfromtxt(csv_file, delimiter = ',')
    frame_nums = data[:,0]
    steering_angles = data[:,1]

    # You could import your images one at a time or all at once first, 
    # here's some code to import a single image:
    frame_num = int(frame_nums[0])
    dim = (32, 32)
    # im_full = cv2.imread(path_to_images + '/' + str(int(frame_num)).zfill(4) + '.jpg')
    input = []
    for i in range(frame_nums.shape[0]):
        im_full = cv2.imread(path_to_images + '/' + str(int(i)).zfill(4) + '.jpg')
        resized = cv2.resize(im_full, dim, interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        im_grey = np.asarray(gray)
        for w in range(256):
            for h in range(256):
                scaled_w=int(w*im.shape[0]/256)
                scaled_h=int(h*im.shape[0]/256)
                scaled_im[w][h]=im_grey[scaled_w][scaled_h]
                G = sobel(scaled_im)
                G_magnitude = np.sqrt(G[0]**2+G[1]**2)
                G_direction = np.arctan2(G[0], G[1])
                thresh = findMean(G_magnitude)
                edges_and_angles = np.zeros(G_magnitude.shape)*np.NaN
                edges_and_angles[G_magnitude > thresh] = G_direction[G_magnitude > thresh]
                edges_and_angles = edges_and_angles[~np.isnan(edges_and_angles)]
                
        
        input.append(edges_and_angles)
    # im_full = cv2.imread(path_to_images + '/' + str(int(frame_num)).zfill(4) + '.jpg',arget_size=(28,28,1), grayscale=True)
    # im_full = im_full/255.
    
    # Train your network here. You'll probably need some weights and gradients!
    
    NN = NeuralNetwork(Lambda=0.0001)
    T =trainer(NN)
    
#     X = np.vstack((im_full.ravel(), ball.ravel()))
#     X = np.array(im_full).reshape(-1, im_shape[0]*im_shape[1])
    
    # grey = np.mean(im_full/255., axis = 2)
#     T.train(np.mean(im_full, axis = 2), steering_angles)
    # im = np.array(im_full.resize(32, 32)) / 255.0
    # dim = (32, 32)
    # resized = cv2.resize(im_full, dim, interpolation = cv2.INTER_AREA)
    # gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # gray = gray-np.mean(gray, axis = 0)
    # cov = np.dot(gray.T, gray) / gray.shape[0]
    # U,S,V = np.linalg.svd(cov)
    # Xrot = np.dot(X, U)
    # Xrot_reduced = np.dot(X, U[:,:100])
    # Xwhite = Xrot / np.sqrt(S + 1e-5)
    # gray = gray/np.std(gray, axis = 0)
    
#     T.train(np.array(input).reshape(-1, dim[0]*dim[1]), steering_angles)
    T.train(np.array(input), steering_angles)
    
    params = NN.getParams()
#     grads = NN.computeGradients(X, y)

    
    
    return NN







def predict(NN, image_file):
    '''
    Second method you need to complete. 
    Given an image filename, load image, make and return predicted steering angle in degrees. 
    '''
    im_full = cv2.imread(image_file)

    ## Perform inference using your Neural Network (NN) here.
    

    return 0.0

class NeuralNetwork(object):
    def __init__(self, Lambda=0):        
        '''
        Neural Network Class, you may need to make some modifications here!
        '''
        self.inputLayerSize = 1
        self.outputLayerSize = 1
        self.hiddenLayerSize = 2
        
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

        options = {'maxiter': 200, 'disp' : True}
#         _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS',args=(trainX, trainY), options=options, callback=self.callbackF)
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS',args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res   
    
    