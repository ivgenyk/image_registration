import cv2
import numpy as np
from matplotlib import pyplot as plt
from LucasKanade import LucasKanade
from brute_force import BruteForce
add_noise = True
show_figs = False
smooth_with_gaus = False

im1 = cv2.imread("im1.jpg")
if smooth_with_gaus:
    n=11
    sig = 3
    im1 = cv2.GaussianBlur(im1, (n, n), sig)

im2 = im1.copy()

# registrator = LucasKanade()
registrator = BruteForce()

for i in range(10):
    random_shift = cv2.randu(np.array((2,1),dtype='float'), -10,10)
    tx, ty = random_shift.tolist()

    M = np.asarray([[1, 0 , tx],[0,1,ty]])
    im2_shifted = cv2.warpAffine(im2, M=M, dsize=None, flags=cv2.INTER_CUBIC,  borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    if add_noise:
        noise = np.zeros_like(im1, dtype='int')
        noise = cv2.randn(noise, np.asarray([0,0,0]) ,np.asarray([30,30,30]))
        im2_shiifted = np.clip(im2_shifted.astype('int') + noise , 0, 255 ).astype('uint8')

    if show_figs:
        plt.figure(1)
        plt.imshow(im1)
        plt.title('original im')
        plt.figure(2)
        plt.imshow(im2_shifted)
        plt.title('shifted tx: %1.3f ty: %1.3f' % (tx,ty))
        plt.show()

    estx, esty = registrator.find_shift(im1, im2_shifted)
    estxp, estyp = registrator.find_shift_pyramid(im1, im2_shifted)

    print('actual shift is: (%1.3f, %1.3f)' % (-tx, -ty))
    print('calculated shift is: (%1.3f, %1.3f)' % (estx, esty))
    print('calculated shift is: (%1.3f, %1.3f)' % (estxp, estyp))
    error = [abs((estx+tx)/tx) ,abs((esty+ty)/ty)]
    error_pyr = [abs((estxp+tx)/tx) ,abs((estyp+ty)/ty)]

    print('error reg',error)
    print('error reg pyramid', error_pyr)
temp = 1