import datetime
import matplotlib.pyplot as plt
#import sys
from Data_Loader import DataLoader
import numpy as np
import os
from glob import glob
from PIL import Image
import imageio
from keras.models import load_model
from srgan import SRGAN

model_name = 'save_model/srgan_gen_final.h5'
save_path = '/home/tcl/Data/ADC/T4/preprocessing/test/'
folder_name = 'LR'

def load_data(dataset_name, img_res=(256, 256),batch_size=1, r=4):
    batch_images = glob('/home/tcl/Data/ADC/T4/preprocessing/%s/*' % (dataset_name))

    #batch_images = np.random.choice(path, size=batch_size)

    imgs_hr = []
    imgs_lr = []
    for img_path in batch_images:
        img = np.array(Image.open(img_path))  # imread(img_path)

        h, w = img_res
        low_h, low_w = int(h / r), int(w / r)

        img_hr = np.array(Image.fromarray(img).resize(img_res))  # scipy.misc.imresize(img, self.img_res)
        img_lr = np.array(Image.fromarray(img).resize((low_h, low_w)))  # scipy.misc.imresize(img, (low_h, low_w))
        print('HR:', img_hr.shape, img_hr.dtype)
        print('LR:', img_lr.shape, img_lr.dtype)

        imgs_hr.append(img_hr)
        imgs_lr.append(img_lr)

    imgs_hr = np.array(imgs_hr) / 127.5 - 1.
    imgs_lr = np.array(imgs_lr) / 127.5 - 1.

    return imgs_hr, imgs_lr

def plotandsave(lr_imgs, fake_imgs):
    # Save generated images and the high resolution originals
    # r= 1
    # c = 1
    # titles = ['Generated', 'Original']
    # fig, axs = plt.subplots(r, c)
    # cnt = 0
    # for row in range(r):
    #     for col, image in enumerate([fake_img, hr_img]):
    #         axs[row, col].imshow(image[row])
    #         axs[row, col].set_title(titles[col])
    #         axs[row, col].axis('off')
    #     cnt += 1
    # fig.savefig("images/%d.png" % (index))
    # plt.close()

    # Save low resolution images for comparison
    index = 0
    for fk in fake_imgs:
        # Save fake higher resolution images for comparison
        #print('fake image type:',fk.shape, fk.dtype)
        #print(fk)
        imageio.imwrite(save_path+'images_fake%d.png' % (index),fk)
        #image = Image.fromarray((fk*255))
        #image.show()
        #image.save(save_path+'images_fake%d.png' % (index))
        #fig = plt.figure()
        #plt.imshow(fk)
        #fig.savefig(save_path+'images_fake%d.png' % (index))
        #plt.close()
        index+=1
    id = 0
    for lr in lr_imgs:
        imageio.imwrite(save_path + 'images_lr%d.png' % (id),lr)
        id += 1

    print("Total save fake and lr:", index, id)

    # Save higher resolution images for comparison
    #fig = plt.figure()
    #plt.imshow(hr_img)
    #fig.savefig(save_path+'images_HR%d.png' % (index))
    #plt.close()
def plotandsaveHR(index, hr_img):
    # Save generated images and the high resolution originals
    # r= 1
    # c = 1
    # titles = ['Generated', 'Original']
    # fig, axs = plt.subplots(r, c)
    # cnt = 0
    # for row in range(r):
    #     for col, image in enumerate([fake_img, hr_img]):
    #         axs[row, col].imshow(image[row])
    #         axs[row, col].set_title(titles[col])
    #         axs[row, col].axis('off')
    #     cnt += 1
    # fig.savefig("images/%d.png" % (index))
    # plt.close()
    # Save higher resolution images for comparison
    #image = Image.fromarray(hr_img.astype('uint8'))
    # image.show()
    imageio.imwrite(save_path + 'images_hr%d.png' % (index),hr_img)



def _main():

    hr_imgs,lr_imgs = load_data(folder_name)
    print('load HR images:',len(hr_imgs))
    print('load LR images:', len(lr_imgs))
    #gan = SRGAN()
    #gen = gan.build_generator()
    model = load_model(model_name)
    print("load model")

    index = 0

    fake_imgs = model.predict(lr_imgs)
    plotandsave(lr_imgs, fake_imgs)

    id = 0
    for hr_img in hr_imgs:
        plotandsaveHR(id, hr_img)
        id+=1
    print("save HR images")



if __name__ == '__main__':
    _main()

