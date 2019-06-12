## Embiggen module: https://github.com/lfsimoes/probav
from embiggen import *
import os
from sklearn.externals import joblib
from glob import glob
from zipfile import ZipFile
import pickle
import numpy as np
import skimage
import cv2
from urllib.request import urlretrieve
from zipfile import ZipFile


''' Preprocessing script; run before modeling '''
DATA_PATH = './probav_data/'
visualize = False
download = True
single_scene = True

''' Variables above:

    DATA_PATH is the path to yout probav folder
    visualize is whether to save obscuration interpolation results
    download True if you do not already have the probav data
    single_scene if to use aggregate for training as opposed to every single image in a scene

    How the data will be set up:

    We have lr and hr train and test

    lr gets saved into a single pickled dictionary ./lr.pickle
    the hr data has to be split due to size into ./x_train_hr.pickle and ./x_test_hr.pickle
'''

def first_process(DATA_PATH, visualize = False, download = True, single_scene = True):

    if download = True:
        urlretrieve('https://kelvins.esa.int/media/competitions/proba-v-super-resolution/probav_data.zip',
                filename='probav_data.zip')
        ZipFile('probav_data.zip').extractall('probav_data/')

    train = all_scenes_paths(DATA_PATH + 'train')
    test  = all_scenes_paths(DATA_PATH + 'test')

    print("##############################")
    print("Length of Train - Test - Total")
    print(len(train), len(test), len(train) + len(test))
    print("##############################")

    median_list = []
    i = 0
    unobscured_lr = []
    high_res = []

    for scene in train:
        print(i)
        scene_path = scene + "/"
        h = cv2.imread(scene_path + 'HR.png',0)

        images = []
        imgs = []
        obsc = []
        for f in glob(scene_path + 'LR*.png'):
            q = f.replace('LR', 'QM')
            l = cv2.imread(f,0)
            c = cv2.imread(q,0)
            images.append((l,c))

        # track obscured pixels
        for (l, c) in images:

            obsc.append(c)
            imgs.append(l)

        agg_opts = {
    		'mean'   : lambda i: np.nanmean(i, axis=0),
    		'median' : lambda i: np.nanmedian(i, axis=0),
    		'mode'   : lambda i: scipy.stats.mode(i, axis=0, nan_policy='omit').mode[0],
    		}

        agg = agg_opts['median']
        agg_img = agg(imgs)

        if single_scene == True:
            print("DOING AGG SCENE")
            unobscured_lr.append(agg_img/255)
            high_res.append(h/255)
            continue

        plot_im = []
        plot_mask = []
        for k in range(len(imgs)):
            img = imgs[k].flatten()
            mask = obsc[k].flatten()
            mean_img = agg_img.flatten()
            img[mask==0] = mean_img[mask==0]

            plot_im.append(img.reshape(128,128))
            plot_mask.append(mask.reshape(128,128))
            img = img.reshape(128,128)
            img = img/255

            unobscured_lr.append(img)
            high_res.append(h/255)

        if visualize == True:
            if i % 50 == 0:
                fig = plt.figure(figsize=(11,11))
                ax1 = fig.add_subplot(331); ax1.imshow(imgs[0]); ax1.axis('off'); ax1.set_title('LR Image 1')
                ax2 = fig.add_subplot(332); ax2.imshow(imgs[1]); ax2.axis('off'); ax2.set_title('LR Image 2')
                ax3 = fig.add_subplot(333); ax3.imshow(imgs[2]); ax3.axis('off'); ax3.set_title('LR Image 3')
                ax4 = fig.add_subplot(334); ax4.imshow(plot_mask[0]); ax4.axis('off'); ax4.set_title('Obscurations Image 1')
                ax5 = fig.add_subplot(335); ax5.imshow(plot_mask[1]); ax5.axis('off'); ax5.set_title('Obscurations Image 2')
                ax6 = fig.add_subplot(336); ax6.imshow(plot_mask[2]); ax6.axis('off'); ax6.set_title('Obscurations Image 3')
                ax7 = fig.add_subplot(337); ax7.imshow(plot_im[0]); ax7.axis('off'); ax7.set_title('Cleaned Image 1')
                ax8 = fig.add_subplot(338); ax8.imshow(plot_im[1]); ax8.axis('off'); ax8.set_title('Cleaned Image 2')
                ax9 = fig.add_subplot(339); ax9.imshow(plot_im[2]); ax9.axis('off'); ax9.set_title('Cleaned Image 3')

                plt.tight_layout()
                plt.savefig("./plots/" + str(i) + "set_clean.png")

        median_list.append(agg_img)

        pickle.dump( median_list, open( "merged_images.pickle", "wb" ) )
        i+=1

    del median_list
    print("NUMBER OF LR IMAGES ",len(unobscured_lr))
    print("NUMBER OF HR ",len(high_res))
    training_data = {"LR":unobscured_lr, "HR":high_res }
    joblib.dump(training_data, "training_data.pickle")


def prepare_and_normalize():
    # read data
    from sklearn.externals import joblib

    all_files = joblib.load('training_data.pickle')

    lr = np.asarray(all_files["LR"])
    hr = np.asarray(all_files["HR"])
    del all_files
    np.random.seed(10)

    # SHUFFLE ALL
    shuf = np.random.shuffle(list(range(0,len(lr))))
    lr = lr[shuf,:,:][0]
    hr = hr[shuf,:,:][0]
    print("[INFO] Shuffled Data !!")
    del shuf

    # SPLIT INTO TEST AND TRAIN
    number_of_train_images = int(len(lr) * 0.8)

    x_train_lr = lr[:number_of_train_images]
    x_train_hr = hr[:number_of_train_images]
    x_test_lr = lr[number_of_train_images:]
    x_test_hr = hr[number_of_train_images:]
    print("Training Length: ", len(lr))
    del lr, hr

    lr = {"x_train_lr":x_train_lr,"x_test_lr":x_test_lr}
    pickle.dump( lr, open( "lr.pickle", "wb" ) )
    del x_train_lr,x_test_lr, lr

    joblib.dump(x_train_hr, 'x_train_hr.pickle')

    del x_train_hr

    joblib.dump(x_test_hr, 'x_test_hr.pickle')
    del x_test_hr


first_process()
prepare_and_normalize()
