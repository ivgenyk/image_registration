import cv2
from matplotlib import pyplot as plt
import numpy as np


def putTextWithBabkground(img, text, text_offset_xy, font=cv2.FONT_HERSHEY_PLAIN, font_scale=1, color=(0, 0, 0),
                          thickness=1):
    # set the text start position
    text_offset_x, text_offset_y = text_offset_xy
    # set the rectangle background to white
    rectangle_bgr = (255, 255, 255)

    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
    if img.shape[0] > box_coords[1][0] and img.shape[1]> box_coords[1][1]:
        cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
        cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=color, thickness=thickness)


class ManualReg:
    """
    Manually shift one image in comparison to the other on and show diff.
    Example:
        im1 = cv2.imread('im1_path.jpg')
        im2 = cv2.imread('im2_path.jpg')
        ManualReg(im1,im2)
    """

    @staticmethod
    def help_str():
        return '-------Help for ManualReg ----------------\n' + \
               'esc: exit \n' + \
               'arrow keys or wdxa keys: move image \n' + \
               '+-: change step size \n' + \
               '? : help \n' + \
               'b : blink between 2 images \n' + \
               'c : switch between color and grayscale \n' + \
               'o : overlay in different colors of the 2 images \n'

    def __init__(self, im1, im2, fig_name='diff im', shift_yx=[0, 0]):
        # # TODO: add image symmetric padding for small images
        # if any([im1.shape[0]<256, im1.shape[1]<256]):
        #     by,bx = max(0,(256 - im1.shape[0])//2), max(0,(256 - im1.shape[1])//2)
        #     im1 = cv2.copyMakeBorder(im1, by, by, bx, bx, cv2.BORDER_CONSTANT, value=(0,0,0)) #top, bottom, left, right
        #     im2 = cv2.copyMakeBorder(im2, by, by, bx, bx, cv2.BORDER_CONSTANT, value=(0,0,0)) #top, bottom, left, right
        # TODO: add support for different size images (align center and not corner)

        self.im1_const = im1
        self.im2_shift = im2
        self.shift = shift_yx
        self.fig_name = fig_name
        self.params = {'step size': 1.,
                       'gray': False}
        self.diplay_func = self.show_diff
        self.enhance_factor = 1
        self.main_loop()
        # cv2.destroyWindow(self.fig_name)

    def show_diff(self):
        """
        Use self params to display diff image in the figure defined in init
        """
        sy, sx = self.im1_const.shape[0:2]
        M = np.asarray([[1., 0, self.shift[0]], [0, 1., self.shift[1]]])
        shifted_im = cv2.warpAffine(self.im2_shift, M, (sx, sy))
        if self.params['gray']:
            const_im = cv2.cvtColor(self.im1_const, cv2.COLOR_BGR2GRAY)
            shifted_im = cv2.cvtColor(shifted_im, cv2.COLOR_BGR2GRAY)
        else:
            const_im = self.im1_const
        diff_im = np.abs(const_im.astype('int16') - shifted_im.astype('int16'))
        diff_im = np.clip(diff_im * self.enhance_factor, 0,255).astype('uint8')
        disp_str = f"x{self.shift[0]} y{self.shift[1]} step{self.params['step size']}"
        # cv2.putText(diff_im, disp_str, (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        putTextWithBabkground(diff_im, disp_str, (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
        cv2.imshow(self.fig_name, diff_im)

    def show_blink(self):
        """
        Display the orig images one after the other with interval of 0.5 sec for 5 times or until key pressed
        """
        num_blinks = 5
        for i in range(num_blinks * 2):
            if i % 2:
                disp_im = self.im1_const
            else:
                sy, sx = self.im1_const.shape[0:2]
                M = np.asarray([[1., 0, self.shift[0]], [0, 1., self.shift[1]]])
                shifted_im = cv2.warpAffine(self.im2_shift, M, (sx, sy))
                disp_im = shifted_im

            disp_str = f"x{self.shift[0]} y{self.shift[1]} step{self.params['step size']}"
            putTextWithBabkground(disp_im, disp_str, (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
            cv2.imshow(self.fig_name, disp_im)
            stop = cv2.waitKey(500)
            if stop > 0:
                break

    def show_overlay(self):
        sy, sx = self.im1_const.shape[0:2]
        M = np.asarray([[1., 0, self.shift[0]], [0, 1., self.shift[1]]])
        shifted_im = cv2.warpAffine(self.im2_shift, M, (sx, sy))
        const_im = cv2.cvtColor(self.im1_const, cv2.COLOR_BGR2GRAY)
        shifted_im = cv2.cvtColor(shifted_im, cv2.COLOR_BGR2GRAY)

        diff_im = np.stack(
            [const_im, shifted_im, np.zeros_like(const_im)],
            axis=2)
        disp_str = f"x{self.shift[0]} y{self.shift[1]} step{self.params['step size']}"
        putTextWithBabkground(diff_im, disp_str, (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
        cv2.imshow(self.fig_name, diff_im)
        # cv2.waitKey(2000)

    def main_loop(self):
        cv2.namedWindow(self.fig_name, cv2.WINDOW_NORMAL)
        # cv2.setMouseCallback(self.fig_name)
        self.show_diff()
        while True:
            key = cv2.waitKey(0)
            if key == 27:  # Esc key to stop
                break
            else:
                self.key_press_func(key)
        cv2.destroyWindow(self.fig_name)

    def key_press_func(self, key_val):
        """
        List of functions triggered by key press
        :param key_val: value of the key that was pressed as in openCV key_press_func
        :return:
        """
        print(key_val)
        # shift image
        if key_val == 81 or key_val == 97:  # left
            self.shift[0] -= self.params['step size']
        elif key_val == 82 or key_val == 119:  # up
            self.shift[1] -= self.params['step size']
        elif key_val == 83 or key_val == 100:  # right
            self.shift[0] += self.params['step size']
        elif key_val == 84 or key_val == 120:  # down
            self.shift[1] += self.params['step size']
        # step size
        elif key_val == 61:  # = - bigger step size
            self.params['step size'] *= 2
        elif key_val == 45:  # - - smaller step size
            self.params['step size'] *= 0.5
        elif key_val == 63 or key_val == 47:  # ? - help
            print(self.help_str())
        # display
        elif key_val == 98:  # b - blink
            self.show_blink()
            # self.diplay_func = self.show_blink
        elif key_val == 99:  # c - color/ gray
            self.params['gray'] = not (self.params['gray'])
        elif key_val == 111:  # o - overlay
            if self.diplay_func.__func__.__name__ is not self.show_overlay.__func__.__name__:
                self.diplay_func = self.show_overlay
            else:
                self.diplay_func = self.show_diff

        elif key_val == 49:  # 1 diff enhance factor
            self.enhance_factor = 1
        elif key_val == 50:  # 2 diff enhance factor
            self.enhance_factor = 2
        elif key_val == 51:  # 3 diff enhance factor
            self.enhance_factor = 3
        elif key_val == 52:  # 4 diff enhance factor
            self.enhance_factor = 4

        self.diplay_func()

    # def images_show(im_list):
    #     plt.figure()
    #
    #     if type(im_list) == 'list':
    #         np_x, np_y = len(im_list), 1
    #         for plt_ind, im in enumerate(im_list):
    #             plt.subplot((np_x, np_y, plt_ind))
    #             plt.imshow(im)
    #
    #     plt.show()





