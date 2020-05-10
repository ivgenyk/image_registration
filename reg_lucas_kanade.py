import numpy as np
import cv2
from reg import RegistrationMethod


class LucasKanade(RegistrationMethod):
    def __init__(self):
        RegistrationMethod.__init__(self)
        self.num_iters = 5
        self.dark_noise_factor = 5
        self.debug_mode = False
        self.method_name = 'LucasKanade'

    def find_shift(self, im1, im2):
        """
        Calculate the subpixel offset between 2 images.
        min images size >20 pixels in each ax.
        :param im1: image BGR or gray
        :param im2: shifted image BGR or gray
        :return: shiftx, shifty float
        """
        im1 = self.preprocess_im(im1)
        im2 = self.preprocess_im(im2)

        Der = calc_derivatives(im1)

        # masking borders to account for filters edge cases
        valid_mask = np.ones_like(im1)
        valid_mask[:5, :] = 0
        valid_mask[-5:, :] = 0
        valid_mask[:, :5] = 0
        valid_mask[:, -5:] = 0

        A = np.asarray(
            [[np.sum(Der['Dxx'] * valid_mask), np.sum(Der['Dxy'] * valid_mask)],
             [np.sum(Der['Dxy'] * valid_mask), np.sum(Der['Dyy'] * valid_mask)]])
        # add bias towards shift (0,0) important for images with little information
        A += self.dark_noise_factor * np.eye(2) * np.sum(valid_mask)
        tx, ty = .0, .0
        # LK iterations
        for k in range(self.num_iters):
            # Apply last shift and calculate It
            M = np.asarray([[1, 0, tx], [0, 1, ty]])
            im2_shifted = cv2.warpAffine(im2, M=M, dsize=None)

            Der['It'] = im1 - im2_shifted
            Der['Dxt'] = Der['Dx'] * Der['It']
            Der['Dyt'] = Der['Dy'] * Der['It']
            B = -1 * np.asarray([np.sum(Der['Dxt'] * valid_mask), np.sum(Der['Dyt'] * valid_mask)])

            # Calculate Shift, normalizing to account for the un normalized filter in grad
            shift = 8 * np.linalg.lstsq(A, B, rcond=None)[0]

            # possible improvements:
            # check if current step is the same size but oposite sign of prev, then cut step by 0.5
            # Break if step is too small
            # handle rank A <2 (1D pattern)

            tx += shift[0]
            ty += shift[1]

            if self.debug_mode:
                plot_debug(im2_shifted, Der)
                print('iter %d tx %f, ty %f' % (k, tx, ty))

        return tx, ty


def plot_debug(im2_shifted, Der):
    from matplotlib import pyplot as plt

    plt.figure(22)
    plt.imshow(im2_shifted)
    plt.show()
    plt.figure(23)
    for i, key in enumerate(list(Der.keys())):
        plt.subplot('24%d' % i)
        plt.imshow(Der[key])
        plt.title(key)


def calc_derivatives(im):
    """
    Calculate derivatives over an image in x and in y.
    :param im: gray image
    :return: dict with: Dx,Dy,Dxx, Dyy, Dxy
    """
    # if want like matlab need zero pad the image before conv...

    hx = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    hy = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    Dx = cv2.filter2D(im, -1, hx, borderType=cv2.BORDER_REPLICATE)
    Dy = cv2.filter2D(im, -1, hy, borderType=cv2.BORDER_REPLICATE)

    # valid_mask = margins_handling(valid_maskIn, 1, 0);#removing kernel size from valid pixels
    Der = {'Dx': Dx, 'Dy': Dy, 'Dxx': np.square(Dx), 'Dyy': np.square(Dy), 'Dxy': Dx * Dy}
    return Der

