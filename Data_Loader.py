#import scipy
from glob import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class DataLoader():

    datapath = '/home/tcl/Data/ADC/T4/preprocessing/'
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res


    def load_data(self,batch_size=1, is_testing=False, r=4):
        data_type = "train" if not is_testing else "test"

        path = glob('/home/tcl/Data/ADC/T4/preprocessing/%s/*' % (self.dataset_name))

        batch_images = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        for img_path in batch_images:
            img = np.array(Image.open(img_path))#imread(img_path)

            h, w = self.img_res
            low_h, low_w = int(h / r), int(w / r)

            img_hr = np.array(Image.fromarray(img).resize(self.img_res))  # scipy.misc.imresize(img, self.img_res)
            img_lr = np.array(Image.fromarray(img).resize((low_h, low_w)))  # scipy.misc.imresize(img, (low_h, low_w))
            print('HR:', img_hr.shape, img_hr.dtype)
            print('LR:', img_lr.shape, img_lr.dtype)

            # If training => do random flip
            if not is_testing and np.random.rand() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return imgs_hr, imgs_lr

    #def imread(image_path):
    #    return np.array(Image.open(image_path, mode='RGB'))  # scipy.misc.imread(path, mode='RGB').astype(np.float)




