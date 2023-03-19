import os
import cv2
import matplotlib.pyplot as plt
import yaml

def get_path( key ):
    with open('paths.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        return data[key]

images_path = get_path('images_path')
labels_path = get_path('labels_path')

def get_files(  path  ):
    images = (  os.listdir( path + images_path )   )
    labels = (  os.listdir( path + labels_path )   )
    images.sort()
    labels.sort()
    images = [read_images(image, path) for image in images]
    labels = [read_labels(label, path) for label in labels]
    return images, labels

def read_images ( image, path ):
    return cv2.imread( path + images_path + image )

def read_labels ( label, path ):
    file = open(path + labels_path + label, "r")
    str = file.read()
    file.close()
    return str

def display_images( images, nr ):
    for i in range(nr):
        plt.figure()
        plt.imshow(images[i])
        plt.show()