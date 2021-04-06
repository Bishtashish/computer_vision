## ---------------------------- ##
## 
## sample_student.py
##
## Example student submission for programming challenge. A few things: 
## 1. Before submitting, change the name of this file to your firstname_lastname.py.
## 2. Be sure not to change the name of the method below, classify.py
## 3. In this challenge, you are only permitted to import numpy and methods from 
##    the util module in this repository. Note that if you make any changes to your local 
##    util module, these won't be reflected in the util module that is imported by the 
##    auto grading algorithm. 
## 4. Anti-plagarism checks will be run on your submission
##
##
## ---------------------------- ##


import numpy as np
import math
import glob

#It's kk to import whatever you want from the local util module if you would like:
#from util.X import ... 
# %pylab inline 
    # Load Image
# im = imread('../data/easy/brick/brick_1.jpg')
    
    
    
def convert_to_grayscale(im):
    '''
    Convert color image to grayscale.
    Args: im = (nxmx3) floating point color image scaled between 0 and 1
    Returns: (nxm) floating point grayscale image scaled between 0 and 1
    '''
    return np.mean(im, axis = 2)

def filter_2d(im, kernel):
    '''
    Filter an image by taking the dot product of each 
    image neighborhood with the kernel matrix.
    Args:
    im = (H x W) grayscale floating point image
    kernel = (M x N) matrix, smaller than im
    Returns: 
    (H-M+1 x W-N+1) filtered image.
    '''

    M, N = kernel.shape
    H, W = im.shape
    filtered_image = np.zeros((H-M+1, W-N+1), dtype = 'float64')
    
    for i in range(filtered_image.shape[0]):
        for j in range(filtered_image.shape[1]):
            image_patch = im[i:i+M, j:j+N]
            filtered_image[i, j] = np.sum(np.multiply(image_patch, kernel))
            
    return filtered_image
def make_gaussian_kernel(size, sigma):
    '''
    Create a gaussian kernel of size x size. 
    Args: 
    size = must be an odd positive number
    sigma = standard deviation of gaussian in pixels
    Returns: A floating point (size x size) guassian kernel 
    ''' 
    #Make kernel of zeros:
    kernel = np.zeros((size, size))
    
    #Handle sigma = 0 case (will result in dividing by zero below if unchecked)
    if sigma == 0:
        return kernel 
    
    #Helpful for indexing:
    k = int((size-1)/2)
    
    for i in range(size):
        for j in range(size):
            kernel[i, j] = (1/(2*np.pi*sigma**2))*math.exp(-((i-k)**2 + (j-k)**2)/(2*sigma**2))
            
    return kernel




# def classify(im):
#     '''
#     Example submission for coding challenge. 
    
#     Args: im (nxmx3) unsigned 8-bit color image 
#     Returns: One of three strings: 'brick', 'ball', or 'cylinder'
    
#     '''
#     #Let's guess randomly! Maybe we'll get lucky.
#     #     labels = ['brick', 'ball', 'cylinder']
#     #     random_integer = np.random.randint(low = 0, high = 3)

#     #     return labels[random_integer]

#     # -------------------------------------------------------

#     #Display Image
#     # fig = figure(0, (6,6))
#     # imshow(im); title('Wow a dumb Brick!');


#     print(classify(im))
   
def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, ax=None):
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indexes of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indexes by their occurrence
        ind = np.sort(ind[~idel])

    return ind
def findMean(G):
#     return mean(G)
    return np.percentile(G,75)
    
def findMode1(G):
    min_g = min(G)
#     print(" Min "+str(min_g))
    max_g = max(G)
#     print(" Max "+str(max_g))
    avg = min_g + max_g

    G=G-min_g
#     per50 = np.percentile(G,50,axis=0)
#     per50 = np.percentile(G,50)
#     m = mean(G)
#     print("Mean "+str(m))
    item = G>avg
    return item
    
def classify(im):
    Kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    gray = convert_to_grayscale(im/255.)
    Gx = filter_2d(gray, Kx)
    Gy = filter_2d(gray, Ky)
    G_magnitude = np.sqrt(Gx**2+Gy**2)
    G_direction = np.arctan2(Gy, Gx)
    thresh = findMean(G_magnitude)
#     fig.add_subplot(num_rows, num_cols, (i*2)+1)
    direct =G_direction[G_magnitude>thresh]
#     print(" no of edges "+str(len(edges_and_angles)))
    counts, bin_edges = np.histogram(direct, bins=60)
    counts.astype(int)
   
    peak_h = detect_peaks(counts, mpd=10, mph=max(counts)*10/100)
#     peak_w = detect_peaks(counts, mph=max(counts)*10/100, mpd=15)
#     peak_h = [x for x in peak_h if x not in peak_w]
    
    newPeaks = []
#     newWidths = []
    for i in range(len(peak_h)):
        newPeaks.append(counts[peak_h[i]])
        
    length = len(newPeaks)
    labels = ['brick', 'ball', 'cylinder']
    val = 0
    if length>=4 and length <=7 and np.mean(newPeaks)>=max(newPeaks)*4/10 and np.mean(newPeaks)<max(newPeaks)*6/10: val =0
    elif length>=4 and np.mean(newPeaks)>max(newPeaks)*6/10 and np.mean(newPeaks)<=max(newPeaks)*7/10: rval = 1
    else: val=2

    return labels[val]
















