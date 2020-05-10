import cv2
import numpy as np
from lucas_kanade_reg import LucasKanade
from manual_reg import ManualReg

"""
In this example we load a sample image, duplicate it. Shift the duplicate and add noise. 
Then using one of the registration methods in order to estimate the shift.
"""


def test_lucas_kanade(im1, im2_shifted):
    registrator = LucasKanade()
    estx, esty = registrator.find_shift(im1, im2_shifted)
    error = [abs((estx + tx) / tx), abs((esty + ty) / ty)]
    print('actual shift is: (%1.3f, %1.3f)' % (-tx, -ty))
    print('calculated shift is: (%1.3f, %1.3f)' % (estx, esty))
    print('error reg', error)


def test_lucas_kanade_pyramid(im1, im2_shifted):
    registrator = LucasKanade()
    estxp, estyp = registrator.find_shift_pyramid(im1, im2_shifted)
    error_pyr = [abs((estxp + tx) / tx), abs((estyp + ty) / ty)]
    print('actual shift is: (%1.3f, %1.3f)' % (-tx, -ty))
    print('calculated shift is: (%1.3f, %1.3f)' % (estxp, estyp))
    print('error reg pyramid', error_pyr)


if __name__ == "__main__":
    add_noise = True
    show_figs = True

    # created 2 images with random shift + noise
    im1 = cv2.imread("im1.jpg")
    im2 = im1.copy()

    random_shift = cv2.randu(np.array((2, 1), dtype='float'), -10, 10)
    tx, ty = random_shift.tolist()

    M = np.asarray([[1, 0, tx], [0, 1, ty]])
    im2_shifted = cv2.warpAffine(im2, M=M, dsize=None, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))
    if add_noise:
        noise = np.zeros_like(im1, dtype='int')
        noise = cv2.randn(noise, np.asarray([0, 0, 0]), np.asarray([30, 30, 30]))
        im2_shiifted = np.clip(im2_shifted.astype('int') + noise.get(), 0, 255).astype('uint8')

    if show_figs:
        from matplotlib import pyplot as plt

        plt.figure(1)
        plt.imshow(im1)
        plt.title('original im')
        plt.figure(2)
        plt.imshow(im2_shifted)
        plt.title('shifted tx: %1.3f ty: %1.3f' % (tx, ty))
        plt.show()

    # testing Lucas Kande algorithm in single resolution
    test_lucas_kanade(im1, im2_shifted)
    # testing Lucas Kande algorithm with pyramid resize
    test_lucas_kanade_pyramid(im1, im2_shifted)
    # manual registration example, press ? to print keys
    ManualReg(im1, im2_shifted)
