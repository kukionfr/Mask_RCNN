
import os
import skimage
from skimage import io
import numpy as np
import cv2
from joblib import delayed, Parallel


def norm(image):
    image2 = skimage.io.imread(image).astype('float32')
    image2 = cv2.normalize(image2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    center_val = np.mean(image2)
    if image2.ndim != 3:
        # print('grayscale to rgb')
        image2 = skimage.color.gray2rgb(image2)

    np.save(image.replace('imset7','train/norm_imset7').replace('tif','npy'),image2)
    return center_val

src = r'\\10.162.80.6\Kyu_Sync\temp\210115 simulated bead images for deep learning\Dataset\set7\imset7'
images = [os.path.join(src,_) for _ in os.listdir(src) if _.endswith('.tif')]
center = Parallel(n_jobs=-2)(delayed(norm)(image) for image in images)
print('{:.2f},{:.2f},{:.2f},{:.2f}'.format(np.min(center),np.mean(center),np.max(center),np.std(center)))



