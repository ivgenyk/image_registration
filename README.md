# image_registration
Methods for image full and subpixel registration

The project contains 4 different methods for finding the offset between 2 images.
Lucas Kanade
POC
Mutual Information
Correlation

The methods can be called in a single call or with different resolution pyramid steps.

Also contains a utility written in openCV for manual registration.
It allows to overlay the images one on top of the other and shift them in varity of step sizes showing the abs diif.
Help for ManualReg
esc: exit
arrow keys or 'w'/'d'/'x'/'a' move the image
+/-: change step size by factor 2/0.5
?: print help
b: blink between 2 images
c: switch color and grayscale
o: display the overlay of the 2 images in different colors
