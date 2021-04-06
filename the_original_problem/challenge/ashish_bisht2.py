

import numpy as np
import glob 
    
    
def convert_to_grayscale(im):

    return np.mean(im, axis = 2)

def filter_2d(im, kernel):

    M, N = kernel.shape
    H, W = im.shape
    filtered_image = np.zeros((H-M+1, W-N+1), dtype = 'float64')
    
    for i in range(filtered_image.shape[0]):
        for j in range(filtered_image.shape[1]):
            filtered_image[i, j] = np.sum(np.multiply(im[i:i+M, j:j+N], kernel))
            
    return filtered_image

def findMean(G):
#     return mean(G)
    return np.percentile(G,93)

def sobel(gray):
    Gx = filter_2d(gray, Kx)
    Gy = filter_2d(gray, Ky)
    return [Gx, Gy]
labels = ['brick', 'ball', 'cylinder']
val = 0
Kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
def classify(im):
    gray = convert_to_grayscale(im/255.)
    G = sobel(gray)
    G_magnitude = np.sqrt(G[0]**2+G[1]**2)
    G_direction = np.arctan2(G[0], G[1])
    thresh = findMean(G_magnitude)
#     fig.add_subplot(num_rows, num_cols, (i*2)+1)
    direct =G_direction[G_magnitude>thresh]
#     print(" no of edges "+str(len(edges_and_angles)))
    counts, bin_edges = np.histogram(direct, bins=60)
    counts.astype(int)
   
    
    a=np.average(counts)
    ma=max(counts)
    delta =max(counts)*thresh/7
    if a<(ma*31/100)+delta: val =0
    elif a+delta>=ma*46/100: val =1
    else: val =2

    return labels[val]
















