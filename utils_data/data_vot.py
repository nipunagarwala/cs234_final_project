import os
import numpy as np
import cPickle as pickle
import argparse
from collections import defaultdict
import random

def construct_sequences(path, output_path, seq_len=8):
    train_output_dir = os.path.join(output_path, 'data', 'train')
    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)
    construct_sequence(path, train_output_dir, seq_len)
    
def construct_sequence(input_path, output, seq_len):    
    for dirname in os.listdir(input_path):
        gt_path = os.path.join(input_path, dirname, "groundtruth_norm.txt")
        labels = get_gt_labels(gt_path)
        video_len = get_num_frames(os.path.join(input_path, dirname, "info.txt"))
        
        for i in range(1, video_len + 1 - seq_len, seq_len):
            curr_seq = []
            curr_labels = []
            for j in range(i, i+seq_len):
                img_path = os.path.join(input_path, dirname, str(j).zfill(8) + "_ds_norm.npy")
                img = np.load(img_path)
                curr_seq.append(img)
                curr_labels.append(labels[j-1])
            curr_dataset = (curr_seq, curr_labels, len(curr_seq))
            curr_output = os.path.join(output, dirname + "-" + str(i).zfill(8))
            with open(curr_output, "wb") as f:
                pickle.dump(curr_dataset, f)
        
def get_num_frames(info):
    num_frames = 0
    with open(info, "r") as f:
        num_frames = int(f.readline().split(",")[0])
    return num_frames
                
def get_gt_labels(gt_filename):
    labels = []
    with open(gt_filename, "r") as f:
        for line in f:
            split_line = map(float, line.strip().split(","))
            values = split_line[0:2] + split_line[-2:]
            labels.append(values)
    return labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images from each sequence")
    parser.add_argument("corpus_path", type=str, help="Path to corpus")
    parser.add_argument("output_path", type=str, help="output data directory")
    args = parser.parse_args()
    construct_sequences(args.corpus_path, args.output_path)

    #construct_sequences("VOT/label_dev")
