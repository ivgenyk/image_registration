import cv2
import numpy as np
from registration import RegistrationMethod


class BruteForce(RegistrationMethod):
    def __init__(self):
        RegistrationMethod.__init__(self)

    def find_shift(self, im1, im2):
        im1 = self.preprocess_im(im1)
        im2 = self.preprocess_im(im2)
        marg = 10
        im2_cropped = im2[marg:-marg, marg:-marg]
        res = cv2.filter2D(im1, ddepth=-1, kernel=im2_cropped, borderType=cv2.BORDER_CONSTANT)
        # ind = np.argmax(res)
        # max_x, max_y = ind % res.shape[0], ind // res.shape[0]
        y,x = np.where(res == np.max(res))
        shifty = y[0] - im1.shape[0] / 2 #- marg
        shiftx = x[0] - im1.shape[1] / 2 #- marg
        return shiftx, shifty