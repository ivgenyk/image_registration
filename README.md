# image_registration
### Methods for image full and subpixel registration

The project contains 4 different automatic methods for finding the offset between 2 images.
#### Implemented methods:
* Lucas Kanade
* POC
* Mutual Information
* Correlation

The methods can be called in a single call or, with different resolution pyramid steps.

example:
```python
    import cv2    
    from reg_lucas_kanade import LucasKanade

    im1 = cv2.imread("im1.jpg")
    im2 = cv2.imread("im2.jpg")
    registrator = LucasKanade()
    estx, esty = registrator.find_shift(im1, im2)    
```
detailed example in example.py

## ManualReg
A utility written in openCV for manual registration.
It allows to overlay the images one on top of the other and shift them in varity of step sizes showing the abs diif.
####Help for ManualReg
* esc: exit
* arrow keys or 'w'/'d'/'x'/'a' move the image
* +/-: change step size by factor 2/0.5
* ?: print help
* b: blink between 2 images
* c: switch color and grayscale
* o: display the overlay of the 2 images in different colors

example:
```python
    import cv2
    from manual_reg import ManualReg

    im1 = cv2.imread("im1.jpg")
    im2 = cv2.imread("im2.jpg")
    ManualReg(im1, im2)
```
detailed example in example.py

#### Requirements
* python 3
* openCV
* numpy
* scipy for mi_reg and poc_reg only
