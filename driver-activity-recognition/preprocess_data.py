"""
Load training images and labels of the drivers and preprocess them

"""
import os
import pickle
import glob
import pandas as pd
import numpy as np
import skimage.io as io
from tqdm import tqdm
from scipy.misc import imresize
from sklearn.preprocessing import LabelBinarizer

#TODO: change to absolute path
data_root_path = "../../all/"

# The default image shape is (480, 640, 3)

SCALE = 5
DISPLAY_SAMPLE = False
SAMPLE_SIZE = 500
CLASSES = ["c0 - safe driving",
           "c1 - texting - right",
           "c2 - talking on the phone - right",
           "c3 - texting - left",
           "c4 - talking on the phone - left",
           "c5 - operating the radio",
           "c6 - drinking",
           "c7 - reaching behind",
           "c8-  hair and makeup",
           "c9 - talking to passenger"]


def one_hot_encode(y):
    lb = LabelBinarizer().fit(range(10))
    return lb.transform(y)


def scale(img):
    return imresize(img, (img.shape[0] // SCALE, img.shape[1] // SCALE))


def normalize(img):
    return img.astype(float) / img.max()


def load_dump_test_data():
    paths = glob.glob(os.path.join(data_root_path, 'imgs/test/*.jpg'))[0:500]
    x_test = []
    y_test = []
    for single_img_path in tqdm(enumerate(paths), total=len(paths)):
        img = io.imread(single_img_path[1])
        img = scale(img)
        img = normalize(img)
        x_test.append(img)

    x_test = np.array(x_test)

    if DISPLAY_SAMPLE:
        io.imshow_collection(x_test)
        io.show()

    with open("test_data.pkl", 'wb') as f:
        pickle.dump((x_test), f)




def load_dump_training_data():
    training_imgs_list = pd.read_csv(os.path.join(data_root_path, 'driver_imgs_list.csv'))
    print(list(training_imgs_list))
    print(training_imgs_list[1000:1020])
    number_of_classes = len(training_imgs_list.classname.value_counts())
    grouped_training_img_list = training_imgs_list.groupby("classname")
    print(grouped_training_img_list.describe())
    x_train = []
    y_train = []

    for class_number in range(number_of_classes):
        training_group = grouped_training_img_list.get_group('c{}'.format(class_number))
        group_path = os.path.join(data_root_path, 'imgs/train/c{}/'.format(class_number)) + training_group.img
        paths = group_path.tolist()[0:SAMPLE_SIZE]
        # sanity check
        # print("number of image files for class {} is {}".format(class_number, len(group_path)))

        for single_img_path in tqdm(enumerate(paths), total=len(paths)):
            img = io.imread(single_img_path[1])
            img = scale(img)
            img = normalize(img)
            x_train.append(img)
            y_train.append(class_number)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    #
    # print("sample image data: shape[{}] max [{}] min[{}]".format(img.shape, img.max(),
    #                                                              img.min()))
    if DISPLAY_SAMPLE:
        io.imshow_collection(x_train)
        io.show()

    with open("preprocessed_data.pkl", 'wb') as f:
        pickle.dump((x_train, y_train), f)


def main():
    load_dump_training_data()
    # load_dump_test_data()

if __name__ == "__main__":
    main()
