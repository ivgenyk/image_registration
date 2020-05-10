import cv2
import numpy as np


class RegistrationMethod:
    def __init__(self):
        self.gauss_smooth ={'n':7, 'sig':3}
        self.method_name = 'anonymous'
        pass

    def find_shift(self, im1, im2):
        """
        find shift required so im2 will be registered to im1
        :param im1:
        :param im2:
        :return:
        """
        shiftx, shifty = float(), float()
        return shiftx, shifty

    def preprocess_im(self, im):
        if np.ndim(im) > 2 and im.shape[2] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = im.astype('float32')
        n, sig = self.gauss_smooth['n'],self.gauss_smooth['sig']
        if n>1:
            im = cv2.GaussianBlur(im, (n, n), sig)
        return im

    def find_shift_pyramid(self, im1, im2, levels=[0.5, 1, 2]):
        shift_x, shift_y = 0., 0.
        for level in levels:
            im1_rsz = cv2.resize(im1, None, fx=level, fy=level, interpolation=cv2.INTER_CUBIC)
            M = np.asarray([[1., 0., shift_x], [0., 1., shift_y]])
            im2_shifted = cv2.warpAffine(im2, M=M, dsize=None, flags=cv2.INTER_CUBIC)
            im2_rsz = cv2.resize(im2_shifted, None, fx=level, fy=level, interpolation=cv2.INTER_CUBIC)
            shift_x_rsz, shift_y_rsz = self.find_shift(im1_rsz, im2_rsz)
            shift_x, shift_y = shift_x + shift_x_rsz / level, shift_y + shift_y_rsz / level
        return shift_x, shift_y