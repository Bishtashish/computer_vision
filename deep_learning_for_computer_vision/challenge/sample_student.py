## ---------------------------- ##
##
## sample_student.py
## Example student submission code for deep learning programming challenge. 
## You are free to use fastai, pytorch, opencv, and numpy for this challenge.
##
## Requirements:
## 0. Make sure your fastai model file is named export.pkl
## 1. Your code must be able to run on CPU or GPU.
## 2. Must handle different image sizes. 
## 3. Use a single unified pytorch model for all 3 tasks. 
## 
## ---------------------------- ##

# from fastai.vision import load_learner, normalize, torch
from fastai.vision import *
import numpy as np

class Model(object):
    def __init__(self, path='../sample_models', file='export.pkl'):
        
        self.learn=load_learner(path=path, file=file) #Load model
        self.class_names=['brick', 'ball', 'cylinder'] #Be careful here, labeled data uses this order, but fastai will use alphabetical by default!

    def predict(self, x):
        '''
        Input: x = block of input images, stored as Torch.Tensor of dimension (batch_sizex3xHxW), 
                   scaled between 0 and 1. 
        Returns: a tuple containing: 
            1. The final class predictions for each image (brick, ball, or cylinder) as a list of strings.
            2. Upper left and lower right bounding box coordinates (in pixels) for the brick ball 
            or cylinder in each image, as a 2d numpy array of dimension batch_size x 4.
            3. Segmentation mask for the image, as a 3d numpy array of dimension (batch_sizexHxW). Each value 
            in each segmentation mask should be either 0, 1, 2, or 3. Where 0=background, 1=brick, 
            2=ball, 3=cylinder. 
        '''

        #Normalize input data using the same mean and std used in training:
        x_norm=normalize(x, torch.tensor(self.learn.data.stats[0]), 
                            torch.tensor(self.learn.data.stats[1]))

    
        #Pass data into model:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            yhat=self.learn.model(x_norm.to(device))
            yhat1=yhat[0]
            yhat2=yhat[1]
            yhat3=yhat[2]

        #Post-processing/parsing outputs, here's an example for classification only:
        class_prediction_indices=yhat1.argmax(dim=1)
        class_predictions=[self.learn.data.classes[i] for i in class_prediction_indices]
        
        # bbox=yhat2
        yhat2=yhat2.detach().cpu()
        # print(yhat2)
        bbox=np.array(yhat2)
        yhat3=yhat3.detach().cpu()
        mask=np.array(yhat3.argmax(dim=1))

        return (class_predictions, bbox, mask)

class MyLoss(nn.Module):
  def forward(self, yhat, y):
    y=y.cpu()
    class_labels=torch.tensor([np.unique(y[i][y[i]!=0])[0] for i in range(y.shape[0])])
    bboxes=torch.zeros((y.shape[0], 4))
    for i in range(y.shape[0]):
      rows,cols= np.where(y[i, 0]!=0)
      bboxes[i, :] = torch.tensor([rows.min(), cols.min(), rows.max(), cols.max()])
    cls_loss=nn.CrossEntropyLoss()(yhat[0],class_labels.to('cuda'))
    det_loss=nn.L1Loss()(yhat[1], bboxes.to('cuda'))
    y=y.to('cuda')
    seg_loss = nn.CrossEntropyLoss()(yhat[2],y.squeeze(dim=1))
    return (1*det_loss) + (10*cls_loss) + (1*seg_loss)


def my_accuracy(yhat, y):
    y=y.cpu()
    class_labels=torch.tensor([np.unique(y[i][y[i]!=0])[0] for i in range(y.shape[0])])
    class_labels = class_labels.to('cuda')
    return accuracy(yhat[0], class_labels.view(-1))

def my_l1(yhat, y):
    y=y.cpu()
    bboxes=torch.zeros((y.shape[0], 4))
    for i in range(y.shape[0]):
      rows,cols= np.where(y[i, 0]!=0)
      bboxes[i, :] = torch.tensor([rows.min(), cols.min(), rows.max(), cols.max()])
    return nn.L1Loss()(yhat[1], bboxes.to('cuda'))

# def pixel_accuracy(yhat, y):
#     y_=y.squeeze(dim=1)
#     # yhat=np.array(yhat)
#     yhat_=yhat[2].argmax(dim=1)
#     return (y_==yhat_).sum().float()/y.numel()

def pixel_accuracy(yhat, y):
    y_=y.squeeze(dim=1)
    yhat_=yhat[2].argmax(dim=1)
    return (y_==yhat_).sum().float()/y.numel()


def conv_trans(ni, nf, ks = 4, stride = 2, padding = 1):
    return nn.Sequential(
        nn.ConvTranspose2d(ni, nf, kernel_size=ks, bias=False, stride=stride, padding = padding), 
        nn.ReLU(inplace = True), 
        nn.BatchNorm2d(nf))

class CustomHead(nn.Module):

  def __init__(self):

    super().__init__()
    
    self.clf = nn.Sequential(
    AdaptiveConcatPool2d((4,4)),
    Flatten(),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(16384,256), #2*512*4*4
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.5),
    nn.Linear(256,4))

    self.det = nn.Sequential(
    AdaptiveConcatPool2d((4,4)),
    Flatten(),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(16384,256), #2*512*4*4
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.5),
    nn.Linear(256,4))
    
    self.seg = nn.Sequential(
    conv_trans(512, 256), 
    conv_trans(256, 128),
    conv_trans(128, 64),
    conv_trans(64, 32), 
    nn.ConvTranspose2d(32, 4, kernel_size=4, bias=False, stride=2, padding = 1))

  def forward(self, x):
    return self.clf(x), self.det(x), self.seg(x)


