"""
=====================
Circular Thresholding
=====================

Circular thresholding is a special case of thresholding for circular signals
(e.g. hue) to create a binary image from a grayscale image [1]_.

.. [1] https://en.wikipedia.org/wiki/Circular_thresholding
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.color.colorconv import hsv2rgb
from skimage.filters import threshold_circular_otsu, threshold_otsu

######################################################################
# We illustrate how to apply one of these thresholding algorithms.
# Otsu's method [2]_ calculates an "optimal" threshold (marked by a red line in the
# histogram below) by maximizing the variance between two classes of pixels,
# which are separated by the threshold. Equivalently, this threshold minimizes
# the intra-class variance.
#
# .. [2] https://en.wikipedia.org/wiki/Otsu's_method
# https://users.cs.cf.ac.uk/Yukun.Lai/papers/thresholdingTIP.pdf


mask = np.fromfunction(lambda r, c: (r - 32) ** 2 + (c - 32) ** 2 < 300, (65, 65))

fig, ax = plt.subplots(5, 3, figsize=(10, 10))
for i in range(5):
    img_hsv = np.ones((*mask.shape, 3), dtype=np.float32)
    hue = img_hsv[..., 0]
    hue[...] = np.where(mask, 0.8, 0.9)
    hue += 0.05 * i
    hue += np.random.normal(0, 0.03, mask.shape)
    hue %= 1.0
    img_rgb = hsv2rgb(img_hsv)

    ax[i, 0].imshow(img_rgb)
    ax[i, 0].axis("off")

    c, x = np.histogram(hue, 256, (0, 1))
    t = threshold_circular_otsu(hue, val_range=(0, 1))
    # equivalent:
    # t = threshold_circular_otsu(val_range=(0, 1), hist=c)
    for v in t:
        ax[i, 1].axvline(v, c="#f00f", lw=2)
    ax[i, 1].axvline(threshold_otsu(hue), c="#0a0", ls="dashed", lw=2)
    ax[i, 1].plot(0.5 * (x[1:] + x[:-1]), c, color="#000")

    ax[i, 2].imshow((hue < t[0]) | (hue > t[1]), cmap="gray")
    ax[i, 2].axis("off")
plt.tight_layout()
plt.show()


######################################################################
# If you are not familiar with the details of the different algorithms and the
# underlying assumptions, it is often difficult to know which algorithm will give
# the best results. Therefore, Scikit-image includes a function to evaluate
# thresholding algorithms provided by the library. At a glance, you can select
# the best algorithm for your data without a deep understanding of their
# mechanisms.
#

from skimage.filters import try_all_threshold

img = data.page()

fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
plt.show()
