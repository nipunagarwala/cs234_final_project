import os
import numpy as np
import cPickle as pickle
import argparse
from collections import defaultdict
import random

SINGLE_OBJECT = True

def construct_train_sequences(path, output_path, seq_len=8):
    train_path = os.path.join(path, "train")
    train_output_dir = os.path.join(output_path, 'data', 'train')
    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)
    construct_sequence(train_path, train_output_dir, seq_len)

def construct_test_sequences(path, seq_len=8):
    test_path = os.path.join(path, "test")
    test_output_dir = os.path.join('data', 'test')
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)
    construct_sequence(test_path, test_output_dir, seq_len)
    
def construct_sequence(input_path, output, seq_len):
    data = []
    labels = []
    seq_lens = []
    
    for dirname in os.listdir(input_path):
        video_len = get_seq_len(os.path.join(input_path, dirname, "seqinfo.ini"))
        for i in range(1, video_len + 1 - seq_len, seq_len):
            curr_seq = []
            curr_labels = []
            
            for j in range(i, i+seq_len):
                img_path = os.path.join(input_path, dirname, "img1", str(j).zfill(6) + "_ds_norm.npy")
                img = np.load(img_path)
                curr_seq.append(img)
                gt_path = os.path.join(input_path, dirname, "gt", str(j).zfill(6) + ".txt")
                label = get_gt_labels(gt_path)
                curr_labels.append(label)
            curr_output = os.path.join(output, dirname + "-" + str(i).zfill(6))
            curr_labels = get_single_label(curr_labels) if SINGLE_OBJECT else curr_labels
            curr_dataset = (curr_seq, curr_labels, len(curr_seq))
            with open(curr_output, "wb") as f:
                pickle.dump(curr_dataset, f)
    
    #dataset = (data, labels, seq_lens)
    #with open(output, "wb") as f:
    #    pickle.dump(dataset, f)
        
def get_seq_len(seqinfo):
    with open(seqinfo, "r") as f:
        for line in f:
            split_line = line.strip().split("=")
            if split_line[0] == "seqLength":
                return int(split_line[1])
                
def get_gt_labels(gt_filename):
    labels = []
    with open(gt_filename, "r") as f:
        for line in f:
            split_line = line.strip().split(",")
            values = map(float, split_line[1:-1])
            values.append(float(split_line[-1]))
            labels.append(values)
    return labels

def get_single_label(curr_labels):
    item_to_step = defaultdict(list)
    for time_step in range(len(curr_labels)):
        for label in curr_labels[time_step]:
            item_to_step[label[0]].append(label)
    keys = item_to_step.keys()
    random.shuffle(keys)
    for item in keys:
        if len(item_to_step[item]) == len(curr_labels):
            return item_to_step[item]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images from each sequence")
    parser.add_argument("corpus_path", type=str, help="Path to corpus")
    parser.add_argument("output_path", type=str, help="Path to output")
    args = parser.parse_args()
    
    construct_train_sequences(args.corpus_path, args.output_path)
    #construct_test_sequences("MOT17_dev")
