'''
Created on Aug 5, 2015

@author: xapharius
'''
import os
import cv2
import numpy as np
import random
import matplotlib.pylab as plt
from numpy import genfromtxt

wildfire_path = "/media/xapharius/Storage/Documents/wildfire/"

def load_img(ix):
    training_file = wildfire_path + "04-Apr-2014-training.txt"
    names = genfromtxt(training_file, delimiter=',', dtype=str)
    img = cv2.imread(wildfire_path + "training/" +  names[ix][0] + "_0.png", -1)
    return img

def plot_img(img):
    plt.figure()
    plt.imshow(img, cmap="gray")

def get_shapes(group):
    '''
    @param group: "training" or "testing"
    '''
    shapes = []
    names_file = wildfire_path + "04-Apr-2014-"+ group +".txt"
    names = genfromtxt(names_file, delimiter=',', dtype=str)
    for name in names[:,0]:
        img = cv2.imread(wildfire_path + group + "/" +  name + "_0.png", -1)
        shapes.append([img.shape[0], img.shape[1]])
    return np.array(shapes)

def transform_img(ix, func):
    training_file = wildfire_path + "04-Apr-2014-training.txt"
    names = genfromtxt(training_file, delimiter=',', dtype=str)
    img0 = cv2.imread(wildfire_path + "training/" +  names[ix][0] + "_0.png", -1)
    img2 = cv2.imread(wildfire_path + "training/" +  names[ix][0] + "_2.png", -1)
    res = func(img2, img0)
    res = cv2.resize(res, (1024, 512))
    #res = (res - res.min()) / float(res.max())
    plt.figure()
    plt.imshow(res, cmap="gray")

def save_set(set_name, func):
    training_file = wildfire_path + "04-Apr-2014-training.txt"
    names = genfromtxt(training_file, delimiter=',', dtype=str)[:,0]
    training_files = ["training/" + name for name in names]
    testing_file = wildfire_path + "04-Apr-2014-testing.txt"
    names = genfromtxt(testing_file, delimiter=',', dtype=str)[:,0]
    testing_files = ["testing/" + name for name in names]
    all_files = training_files + testing_files

    if not os.path.exists(wildfire_path + set_name):
        os.mkdir(wildfire_path + set_name)

    skipped = 0
    for it, file_name in enumerate(all_files):
        print "{}/{}".format(it+1, len(all_files))
        img0 = cv2.imread(wildfire_path + file_name + "_0.png", -1)
        img2 = cv2.imread(wildfire_path + file_name + "_2.png", -1)
        smoke = cv2.imread(wildfire_path + file_name + "_smoke.png", -1)
        if img0 is None or img2 is None:
            print "skipped"
            skipped += 1
            continue
        res = func(img2, img0)
        #res = (res - res.min()) / float(res.max())
        res = multisect(res, smoke)
        output_path = wildfire_path + set_name + "/" + file_name.split("/")[1]
        #print output_path
        for i, img in enumerate(res):
            cv2.imwrite(output_path + "_" + str(i) + ".png", img)
    print "skipped total:", skipped

def multisect(img, smoke=None):
    """
    Getting a list of 5 512x512 images, sampled randomly starting from
    each quadrant + center
    """
    res = []
    xylen = 512
    max_x = img.shape[1]
    max_y = img.shape[0]
    if smoke is None:
        # random
        wiggle_x = max_x - xylen
        wiggle_y = max_y - xylen
        # upper left
        x = random.randint(0, wiggle_x/2)
        y = random.randint(0, wiggle_y/2)
        res.append(img[y:y + xylen, x:x + xylen])
        assert res[-1].shape == (512, 512), "upper left"
        # lower left
        x = random.randint(0, wiggle_x/2)
        y = random.randint(wiggle_y/2, wiggle_y)
        res.append(img[y:y + xylen, x:x + xylen])
        assert res[-1].shape == (512, 512), "lower left"
        # lower right
        x = random.randint(wiggle_x/2, wiggle_x)
        y = random.randint(wiggle_y/2, wiggle_y)
        res.append(img[y:y + xylen, x:x + xylen])
        assert res[-1].shape == (512, 512), "lower right"
        # upper right
        x = random.randint(wiggle_x/2, wiggle_x)
        y = random.randint(0, wiggle_y/2)
        res.append(img[y:y + xylen, x:x + xylen])
        assert res[-1].shape == (512, 512), "upper right"
        # center
        x = wiggle_x / 2
        y = wiggle_y / 2
        res.append(img[y:y + xylen, x:x + xylen])
        assert res[-1].shape == (512, 512), "center"
        return res

    # find fire coords
    x1, y1, w, h = cv2.boundingRect(smoke)
    x2 = x1 + w
    y2 = y1 + h
    wiggle_y_min = min(max(y2 - xylen, 0), max(y1 - 100, 0))
    wiggle_y_max = max(min(y1 + xylen, max_y), min(y2 + 100, max_y)) - xylen
    wiggle_y_mid = (wiggle_y_min + wiggle_y_max)/2
    wiggle_x_min = min(max(x2 - xylen, 0), max(x1 - 100, 0))
    wiggle_x_max = max(min(x1 + xylen, max_x), min(x2 + 100, max_x)) - xylen
    wiggle_x_mid = (wiggle_x_min + wiggle_x_max)/2
    # upper left
    x = random.randint(wiggle_x_min, wiggle_x_mid)
    y = random.randint(wiggle_y_min, wiggle_y_mid)
    res.append(img[y:y + xylen, x:x + xylen])
    assert res[-1].shape == (512, 512), "upper left"
    # lower left
    x = random.randint(wiggle_x_min, wiggle_x_mid)
    y = random.randint(wiggle_y_mid, wiggle_y_max)
    res.append(img[y:y + xylen, x:x + xylen])
    assert res[-1].shape == (512, 512), "lower left"
    # lower right
    x = random.randint(wiggle_x_mid, wiggle_x_max)
    y = random.randint(wiggle_y_mid, wiggle_y_max)
    res.append(img[y:y + xylen, x:x + xylen])
    assert res[-1].shape == (512, 512), "lower right"
    # upper right
    x = random.randint(wiggle_x_mid, wiggle_x_max)
    y = random.randint(wiggle_y_min, wiggle_y_mid)
    res.append(img[y:y + xylen, x:x + xylen])
    assert res[-1].shape == (512, 512), "upper right"
    # center
    x = wiggle_x_mid
    y = wiggle_y_mid
    res.append(img[y:y + xylen, x:x + xylen])
    assert res[-1].shape == (512, 512), "center"
    return res


"""
#10202_20100727_U_145618_0855 # no smoke
#10202_20091109_U_083111_0855 #smoke 1300x800
#10301_20100415_U_070535_0450 #smoke 1380x768 huge smoke
name = "10301_20100415_U_070535_0450"
img1 = cv2.imread(wildfire_path + "training/" + name + "_0.png", -1)
smoke = cv2.imread(wildfire_path + "training/" + name + "_smoke.png", -1)

plot_img(img1)
imgs = multisect(img1, smoke)
for img in imgs:
    plot_img(img)
plt.show()
"""

"""
save_set("diff", lambda x,y: x-y)
save_set("div", lambda x,y: x/y.astype(float))
save_set("img2", lambda x,y: x)
"""

#0109_20100720_U_124303_1470.png


img1 = cv2.imread(wildfire_path + "training/0101_20100719_U_181948_2680_0.png", -1)
img2 = cv2.imread(wildfire_path + "training/0101_20100719_U_181948_2680_2.png", -1)
plot_img(img2-img1)
plot_img(img2)
plt.show()


"""
transform_img(30, lambda x,y: x-y)
transform_img(30, lambda x,y: x)
transform_img(30, lambda x,y: x/y.astype(float))
plt.show()
"""

"""
img1 = cv2.imread(wildfire_path + "training/0109_20100720_U_124303_1470_0.png", -1) # -1 = "as i is"
plt.figure()
plt.imshow(img1, cmap="gray")
plt.show()
"""


