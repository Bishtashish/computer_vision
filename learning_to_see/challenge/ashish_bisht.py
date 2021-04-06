import numpy as np

def breakIntoGrids(im, s = 9):
    grids = []

    h = s//2 #half grid size minus one.
    for i in range(h, im.shape[0]-h):
        for j in range(h, im.shape[1]-h):
            grids.append(im[i-h:i+h+1,j-h:j+h+1].ravel())

    return np.vstack(grids)

def reshapeIntoImage(vector, im_shape, s = 9):
    h = s//2 #half grid size minus one. 
    image = np.zeros(im_shape)
    image[h:-h, h:-h] = vector.reshape(im_shape[0]-2*h, im_shape[1]-2*h)

    return image

labels = [1, 2, 3]
def countEdges(splicedRow):
    count=0
    for i in range(len(splicedRow)):
        if i==0:
            if splicedRow[i]==True:
                count=count+1   
        else:
            if splicedRow[i-1]==False and splicedRow[i]==True:
                count=count+1
    return count

def count_fingers(im):
    im = im > 94 #Threshold image
    X = breakIntoGrids(im, s = 8) #Break into 9x9 grids
    #Use rule we learned with decision tree
#     treeRule1 = lambda X: np.logical_and(np.logical_and(X[:, 40] == 1, X[:,0] == 0), X[:, 53] == 0)
    treeRule1 = lambda X: np.logical_and(np.logical_and(X[:, 30] == 1, X[:,0] == 0), X[:, 53] == 0)
    yhat = treeRule1(X)
    yhat_reshaped = reshapeIntoImage(yhat, im.shape)
    splicedRow = yhat_reshaped[yhat_reshaped.shape[0]//4,:]
    return countEdges(splicedRow.tolist())
