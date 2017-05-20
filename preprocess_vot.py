import argparse
import numpy as np
import os
from scipy import misc, ndimage
import matplotlib.pyplot as plt
from collections import defaultdict
import math

def get_sizes(path):
    shapes = set()
    min_height = 1080
    min_width = 1920
    for dirpath, dirnames, filenames in os.walk(path):
        if len(filenames) > 0 and filenames[0][-4:] == ".jpg":
            full_path = os.path.join(dirpath, filenames[0])
            img = misc.imread(full_path)
            shapes.add(img.shape)
            min_height = min(img.shape[0], min_height)
            min_width = min(img.shape[1], min_width)
    return min_height, min_width

def process_vot(path, min_height, min_width):
    images = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename[-4:] == ".jpg" and "_ds" not in filename:
                full_path = os.path.join(dirpath, filename)
                img = misc.imread(full_path,mode='RGB')
                ratio = min(float(min_width)/img.shape[1], float(min_height)/img.shape[0])
                img = misc.imresize(img, size=ratio)
                img = pad_image(img, (min_height, min_width))
                output_filename = os.path.join(dirpath, filename[:-4] + "_ds.jpg")
                misc.imsave(output_filename, img)
                images.append(output_filename)
    return images
                
def pad_image(img, pad_size):
    img = img.transpose(2,0,1)
    diff0_before = (pad_size[0] - img.shape[1]) / 2
    diff1_before = (pad_size[1] - img.shape[2]) / 2
    diff0_after = int(math.ceil((pad_size[0] - img.shape[1]) / 2.0))
    diff1_after = int(math.ceil((pad_size[1] - img.shape[2]) / 2.0))
    
    print img.shape, pad_size, diff0_before, diff0_after, diff1_before, diff1_after
    img = np.asarray([np.pad(x, ((diff0_before,diff0_after), (diff1_before,diff1_after)), 'constant', constant_values=(np.median(x) ,)) for x in img])
    return img.transpose(1,2,0)

def training_set_mean_stdev(images, dimensions):
    average_img = np.zeros(dimensions)
    average_sq_img = np.zeros(dimensions)
    num_images = float(len(images))
    for img_path in images:
        img = misc.imread(img_path)
        average_img += img / num_images
        average_sq_img += np.asarray([np.square(x) for x in average_img.transpose(2,0,1)]).transpose(1,2,0) / num_images
    mean = np.asarray([np.mean(x) for x in average_img.transpose(2,0,1)])
    # stdev = np.asarray([np.mean(x) for x in average_sq_img.transpose(2,0,1)]) - np.square(mean)
    stdev = np.asarray([0, 0, 0])
    return mean, stdev    
    
def normalize_training_set(images, mean, stdev, size):
    channel0 = np.ones(size) * mean[0]
    channel1 = np.ones(size) * mean[1]
    channel2 = np.ones(size) * mean[2]
    mean_pixel = np.asarray([channel0, channel1, channel2]).transpose(1,2,0).astype('int8')
    #print mean_pixel.shape
    for img_path in images:
        img = misc.imread(img_path).astype('int8')
        img -= mean_pixel
        output_filename = img_path[:-4] + "_norm"
        np.save(output_filename, img, allow_pickle=False)
        plt.imshow(img, interpolation='none')
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images from each sequence")
    parser.add_argument("dataset_path", type=str, help="Path to dataset")
    args = parser.parse_args()
    path = args.corpus_path
    
    min_height, min_width = get_sizes(path)
    images = process_vot(path, min_height, min_width)
    mean, stdev = training_set_mean_stdev(images, (min_height, min_width, 3))
    normalize_training_set(images, mean, stdev, (min_height, min_width))
