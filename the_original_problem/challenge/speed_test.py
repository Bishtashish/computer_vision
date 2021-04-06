import numpy as np


def convert_to_grayscale(im):
    return np.mean(im, axis=2)

def fftc(im, kernel):
    s1 = np.array(im.shape)
    s2 = np.array(kernel.shape)
    size = s1 + s2 - 1

    fsize = 2 ** np.ceil(np.log2(size)).astype(int)
    fslice = tuple([slice(0, int(sz)) for sz in size])

    new_x = np.fft.fft2(im , fsize)
    new_y = np.fft.fft2(kernel , fsize)
    result = np.fft.ifft2(new_x*new_y)[fslice].copy()
    # return np.array(result.real , np.int32)
    return np.array(result.real , np.float64)


def sobel_x(img):
    result_h = img[:, 2:] - img[:, :-2]
    result_v = result_h[:-2] + result_h[2:] + 2*result_h[1:-1]
    return result_v

def sobel1(im):
    H, W = im.shape
    result_h = np.zeros((H-M+1, W-N+1), dtype='float64')
    result_v = np.zeros((H-M+1, W-N+1), dtype='float64')
    for i in range(0,result_h.shape[1]):
        result_h[:,i:i+1] = im[:, i+2:] - im[:, :i-2]
    for j in range(0,result_h.shape[0]):
        result_v = result_h[:-2] + result_h[2:] + 2*result_h[1:-1]

    return result_v



def findMean(G):
    percentile = np.percentile(G, 96)
    print(percentile)
    if percentile >= 1.0:
        percentile = percentile-1
    if percentile <= 0.10:
        percentile = percentile+.1
    if percentile > 0.95:
        percentile = percentile-.1
    return percentile


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
    G = [None]*2
    G[0] = fftc(gray,Kx)
    G[1] = fftc(gray,Ky)
    G_magnitude = np.sqrt(G[0]**2+G[1]**2)
    # G_magnitude = abs(G[0]) + abs(G[1])
    # G_magnitude*=255./max(G_magnitude)
    G_direction = np.arctan2(G[0], G[1])
    thresh = findMean(G_magnitude)
    edges_and_angles = np.zeros(G_magnitude.shape)*np.NaN
    edges_and_angles[G_magnitude > thresh] = G_direction[G_magnitude > thresh]
    edges_and_angles = edges_and_angles[~np.isnan(edges_and_angles)]
    counts, bin_edges = np.histogram(edges_and_angles, bins=60)
    counts.astype(int)
    a = np.average(counts)
    ma = max(counts)
    delta = ma*thresh/8
    if a < (ma*31/100)+delta:
        val = 0
    elif a+delta >= ma*46/100:
        val = 1
    else:
        val = 2
    return labels[val]
