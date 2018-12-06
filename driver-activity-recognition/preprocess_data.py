"""
Load training images and labels of the drivers and preprocess them

"""
import os
import pandas as pd
import skimage.io as io
from scipy.misc import imresize

#TODO: change to absolute path
data_root_path = "../../all/"

# The default image shape is (480, 640, 3)

SCALE = 10


def load_training_data():
    training_imgs_list = pd.read_csv(os.path.join(data_root_path, 'driver_imgs_list.csv'))
    print(list(training_imgs_list))
    number_of_classes = len(training_imgs_list.classname.value_counts())
    grouped_training_img_list = training_imgs_list.groupby("classname")
    print(grouped_training_img_list.describe())
    x_train = []
    y_train = []

    for i in range(number_of_classes):
        training_group = grouped_training_img_list.get_group('c{}'.format(i))
        group_path = os.path.join(data_root_path, 'imgs/train/c{}/'.format(i)) + training_group.img

        # read a sample of images
        img = io.imread(group_path.tolist()[0])
        img = imresize(img, (img.shape[0]//SCALE, img.shape[1]//SCALE))
        x_train.append(img)
        y_train.append(i)

    io.imshow_collection(x_train)
    io.show()


def main():
    load_training_data()


if __name__ == "__main__":
    main()
