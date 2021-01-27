from PIL import Image
from joblib import Parallel, delayed
import pickle
from time import time
import numpy as np
import os
from pycocotools import mask as maskUtils

def mask_convert_3d(label):
    dim = 256

    mask = Image.open(label)
    mask = np.array(mask).astype(np.int16)
    label_list = np.unique(mask)[1:]

    masklayers = np.ones((dim,dim,len(label_list)))
    sz=[]
    for idx,maskidx in enumerate(label_list):
        masklayers[:,:,idx]=mask==maskidx
        sz.append(np.sum(mask==maskidx))
    masklayersf = np.asfortranarray(masklayers).astype(np.uint8)

    class_ids = [1] * len(label_list)
    rles = maskUtils.encode(masklayersf)

    ann = {'annotation': rles, 'class_ids': class_ids}
    with open(label.replace('tif', 'pkl'), 'wb') as fout:
        pickle.dump(ann, fout, pickle.HIGHEST_PROTOCOL)

    # print('time to generate pkl',np.around(time() - start))
    return [len(label_list),np.min(sz),np.mean(sz),np.max(sz),np.std(sz)]

src = r'\\10.162.80.6\Kyu_Sync\temp\210115 simulated bead images for deep learning\Dataset\set7\imset7_lbl'
tifs = [os.path.join(src,_) for _ in os.listdir(src) if _.endswith('.tif')]
s=time()
objectnum = Parallel(n_jobs=-2)(delayed(mask_convert_3d)(tif) for tif in tifs)
print(time()-s)
print('number of images',len(tifs))
objectnum = np.array(objectnum)
print('number of objects {:.2f},{:.2f},{:.2f},{:.2f}'.format(np.min(objectnum[:,0]),np.mean(objectnum[:,0]),np.max(objectnum[:,0]),np.std(objectnum[:,0])))
print('minimum area {:.2f},{:.2f},{:.2f},{:.2f}'.format(np.min(objectnum[:,1]),np.mean(objectnum[:,1]),np.max(objectnum[:,1]),np.std(objectnum[:,1])))
print('mean area {:.2f},{:.2f},{:.2f},{:.2f}'.format(np.min(objectnum[:,2]),np.mean(objectnum[:,2]),np.max(objectnum[:,2]),np.std(objectnum[:,2])))
print('max area {:.2f},{:.2f},{:.2f},{:.2f}'.format(np.min(objectnum[:,3]),np.mean(objectnum[:,3]),np.max(objectnum[:,3]),np.std(objectnum[:,3])))
print('std area {:.2f},{:.2f},{:.2f},{:.2f}'.format(np.min(objectnum[:,4]),np.mean(objectnum[:,4]),np.max(objectnum[:,4]),np.std(objectnum[:,4])))




