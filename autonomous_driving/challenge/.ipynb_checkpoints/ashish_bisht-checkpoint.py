import numpy as np
import cv2
from tqdm import tqdm
import time
# from scipy import optimize
from scipy.special import expit

max_angle = 180
min_angle = -180
bin_size = 64


def train(path_to_images, csv_file):
    data = np.genfromtxt(csv_file, delimiter = ',')
    frame_nums = data[:,0]
    steering_angles = data[:,1]
    theta = np.matrix(steering_angles).transpose()
    bins = np.linspace(min_angle,max_angle,bin_size) 
    values = np.linspace(0,bin_size -1,bin_size)
    y = np.zeros((len(theta),bin_size))
        
    for i,angle in enumerate(theta):
        index =  int(np.interp(angle,bins, values))
        y[i,index] = 1.0
        
        if index - 1 >= 0: 
            y[i,index - 1] = 0.89
        if index + 1 < bin_size:
            y[i,index + 1] = 0.89
            
            if index - 2 >= 0:  
                y[i,index - 2] = 0.6
            if index + 2 < bin_size:
                y[i,index + 2] = 0.6
                
                if index - 3 >= 0:  
                    y[i,index - 2] = 0.3
                if index + 3 < bin_size:
                    y[i,index + 2] = 0.3  
    X = []
    for i in range(frame_nums.shape[0]):     
        im_full = cv2.imread(path_to_images + '/' + str(int(i)).zfill(4) + '.jpg')
        im_full = im_full[:, :, 2]
        im_full = cv2.resize(im_full, (60,60)) 
        X.append(np.ravel(np.array(im_full[30:,:])/255.0))
    iterations = 3500
    learning_rate = 1e-1*(4)
    
    # Creating an instance of the Neural Network Class
    NN = Neural_Network(Lambda=learning_rate) 
    X = np.reshape(X,(1500,(1800)))
    for iter in tqdm(range(iterations)):    
        # Compute Gradients
        gradients  = NN.computeGradients(X, y)
        # Get the weights
        params = NN.getParams()   
        # Perform gradient descent
        params[:] = params[:] - ((learning_rate*gradients[:])/(len(X)))  
        # Updating weights
        NN.setParams(params)
    return NN



# Predict function
def predict(NN, image_file):
    im_full = cv2.imread(image_file)
    im_full = im_full[:, :, 2]
    im_full = cv2.resize(im_full, (60,60))
    image_vector = np.array(im_full[30:,:])/255.0
    image_vector = np.reshape(image_vector,(1,-1))
    bins = np.linspace(min_angle,max_angle,bin_size) 
    return bins[np.argmax(NN.forward(image_vector))]        

class Neural_Network(object):
    def __init__(self, Lambda=0):        
        self.inputLayerSize = 1800
        self.outputLayerSize = bin_size
        self.hiddenLayerSize = 128
        
        # Inititalize the weights 
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)*np.sqrt(1/(self.inputLayerSize + self.hiddenLayerSize))
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)*np.sqrt(1/(self.outputLayerSize + self.hiddenLayerSize))
        self.Lambda = Lambda


    def forward(self, X):
        # Propogate inputs though network 
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix 
        # return 1/(1+np.exp(-z))
        return expit(z) 
    
    def sigmoidPrime(self,z):
        # Gradient of sigmoid 
        # return np.exp(-z)/((1+np.exp(-z))**2)
        return (expit(z)*(1-expit(z)))
    
    def costFunction(self, X, y):
        # Compute cost for given X,y, use weights already stored in class
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2) + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
        return J
        
    def costFunctionPrime(self, X, y):
        # Compute derivative with respect to W and W2 for a given X and y
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)+ self.Lambda*self.W2
        
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2) + self.Lambda*self.W1
         
        
        return dJdW1, dJdW2
        
   
    def getParams(self):
        # Get W1 and W2 unrolled into vector
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
        
    def setParams(self, params):
        # Set W1 and W2 using single paramater vector
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))