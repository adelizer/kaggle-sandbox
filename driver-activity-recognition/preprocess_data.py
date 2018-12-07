"""
Load training images and labels of the drivers and preprocess them

"""
import os
import pandas as pd
import skimage.io as io
from scipy.misc import imresize
from sklearn.preprocessing import LabelBinarizer

#TODO: change to absolute path
data_root_path = "../../all/"

# The default image shape is (480, 640, 3)

SCALE = 20
DISPLAY_SAMPLE = True
SAMPLE_SIZE = 2


def one_hot_encode(y):
    lb = LabelBinarizer().fit(range(10))
    return lb.transform(y)

def scale(img):
    return imresize(img, (img.shape[0] // SCALE, img.shape[1] // SCALE))


def normalize(img):
    return img.astype(float) / img.max()


def load_training_data():
    training_imgs_list = pd.read_csv(os.path.join(data_root_path, 'driver_imgs_list.csv'))
    print(list(training_imgs_list))
    number_of_classes = len(training_imgs_list.classname.value_counts())
    grouped_training_img_list = training_imgs_list.groupby("classname")
    print(grouped_training_img_list.describe())
    x_train = []
    y_train = []

    for class_number in range(number_of_classes):
        training_group = grouped_training_img_list.get_group('c{}'.format(class_number))
        group_path = os.path.join(data_root_path, 'imgs/train/c{}/'.format(class_number)) + training_group.img

        # sanity check
        print("number of image files for class {} is {}".format(class_number, len(group_path)))

        for single_img_path in group_path.tolist()[0:SAMPLE_SIZE]:
            img = io.imread(single_img_path)
            img = scale(img)
            img = normalize(img)
            x_train.append(img)
            y_train.append(class_number)

    print("sample image data: shape[{}] max [{}] min[{}]".format(img.shape, img.max(),
                                                                 img.min()))
    if DISPLAY_SAMPLE:
        io.imshow_collection(x_train)
        io.show()

    print(y_train)
    print(one_hot_encode(y_train))


def main():
    load_training_data()


if __name__ == "__main__":
    main()
