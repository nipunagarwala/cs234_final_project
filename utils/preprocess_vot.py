import argparse
import numpy as np
import os
from scipy import misc, ndimage
import matplotlib.pyplot as plt
from collections import defaultdict
import math
from operator import add

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
    return shapes, min_height, min_width

def process_vot(path, min_height, min_width):
    images = []
    for dirpath, dirnames, filenames in os.walk(path):
        img_shape = None
        pad_height = 0
        pad_width = 0
        for filename in filenames:
            if filename[-4:] == ".jpg" and "_ds" not in filename:
                full_path = os.path.join(dirpath, filename)
                img = misc.imread(full_path,mode='RGB')
                img_shape = img.shape
                ratio = min(float(min_width)/img.shape[1], float(min_height)/img.shape[0])
                img = misc.imresize(img, size=ratio)
                img, pad_height, pad_width = pad_image(img, (min_height, min_width))
                output_filename = os.path.join(dirpath, filename[:-4] + "_ds.jpg")
                misc.imsave(output_filename, img)
                images.append(output_filename)
        if img_shape:
            gt_path = os.path.join(dirpath, "groundtruth.txt")
            preprocess_label(gt_path, img_shape, min_height, min_width, pad_height, pad_width)
    return images
                
def pad_image(img, pad_size):
    img = img.transpose(2,0,1)
    diff0_before = (pad_size[0] - img.shape[1]) / 2
    diff1_before = (pad_size[1] - img.shape[2]) / 2
    diff0_after = int(math.ceil((pad_size[0] - img.shape[1]) / 2.0))
    diff1_after = int(math.ceil((pad_size[1] - img.shape[2]) / 2.0))
    img = np.asarray([np.pad(x, ((diff0_before,diff0_after), (diff1_before,diff1_after)), 'constant', constant_values=(np.median(x) ,)) for x in img])
    return img.transpose(1,2,0), diff0_before, diff1_before

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
    for img_path in images:
        img = misc.imread(img_path).astype('int8')
        img -= mean_pixel
        output_filename = img_path[:-4] + "_norm"
        np.save(output_filename, img, allow_pickle=False)

def preprocess_label(gt_path, orig_shape, height, width, offset_height, offset_width):
    scale = [orig_shape[1], orig_shape[0]] * 4
    offset = [offset_width/float(width), offset_height/float(height)] * 4
    normalized_lines = ""
    output_file = os.path.join(os.path.dirname(gt_path), "groundtruth_norm.txt")
    info_file = os.path.join(os.path.dirname(gt_path), "info.txt")
    num_lines = 0
    with open(gt_path, "r") as f:
        for line in f:
            split_line = map(float, line.split(",")) 
            if len(split_line) != 8:
                print gt_path
                return
            normalized = map(add, map(lambda x,y: x/y, split_line, scale), offset)
            new_line = ",".join([format(x, "0.6f") for x in normalized]) + "\n"
            normalized_lines += new_line
            num_lines += 1
    with open(output_file, "w") as of:
        of.write(normalized_lines)
    with open(info_file, "w") as info:
        info.write("%s,%s,%s" % (num_lines, orig_shape[0], orig_shape[1]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images from each sequence")
    parser.add_argument("dataset_path", type=str, help="Path to dataset")
    args = parser.parse_args()
    path = args.dataset_path
    
    video_sizes, min_height, min_width = get_sizes(path)
    images = process_vot(path, min_height, min_width)
    mean, stdev = training_set_mean_stdev(images, (min_height, min_width, 3))
    normalize_training_set(images, mean, stdev, (min_height, min_width))
