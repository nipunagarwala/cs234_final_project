import numpy as np 
import sys
import os
import argparse
import cPickle as pickle


def parse_commandline():
    """
    Parses the command line arguments to the run method for training and testing purposes
    Inputs:
        None
    Returns:
        args: An object with the command line arguments stored in the correct values.
            phase : Train or Test
            train_path : Path for the training data
            val_path : Path for the testing data
            save_every : (Int) How often to save the model
            save_to_file : (string) Path to file to save the model too
            load_from_file : (string) Path to load the model from
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', choices=['train', 'test'])
    parser.add_argument('--train_path', nargs='?', default='./data/hw3_train.dat', type=str, help="Give path to training data")
    parser.add_argument('--val_path', nargs='?', default='./data/hw3_val.dat', type=str, help="Give path to val data")
    parser.add_argument('--save_every', nargs='?', default=2, type=int, help="Save model every x iterations. Default is not saving at all.")
    parser.add_argument('--save_to_file', nargs='?', default=os.getcwd()+ '/' + 'checkpoints/model_ckpt', type=str, help="Provide filename prefix for saving intermediate models")
    parser.add_argument('--load_from_file', nargs='?', default=None, type=str, help="Provide filename to load saved model")
    args = parser.parse_args()
    return args

'''
    Given a list/array of video frames and labels, this shuffles the
    list/array and returns 3 lists of numpy arrays, i.e. video frames,
    labels and sequence lengths. Each list contains batches of size
    batch_size corresponding
    to video frames, labels or sequence lengths, depending on the list.
    Each time make_batches is called, the dataset
    sent as input is shuffled and new batches are created.

'''
def make_batches(dataset, batch_size=32):
    orig_data, orig_labels, orig_seq_lens = dataset

    indices = np.random.permutation(len(orig_data))
    data = orig_data[indices]
    labels = orig_labels[indices]
    seq_lens = orig_seq_lens[indices]
    
    batched_data = []
    batched_labels = []
    batched_seq_lens = []
    num_batches = int(np.ceil(len(data) / float(batch_size)))
    
    for i in range(num_batches):
        batch_start = i * batch_size
        batched_data.append(data[batch_start : batch_start + batch_size])
        batched_labels.append(labels[batch_start : batch_start + batch_size])
        batched_seq_lens.append(seq_lens[batch_start : batch_start + batch_size])
    
    return batched_data, batched_labels, batched_seq_lens

'''
    Reads in the videos and corresponding labels into memory and
    returns them respectively.

'''
def load_dataset(dataset_path):
    data = []
    labels = []
    seq_lens = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            with open(os.path.join(dirpath, filename), 'rb') as f:
                curr_seq, curr_labels, seq_len = pickle.load(f)
                data.append(curr_seq)
                labels.append(curr_labels)
                seq_lens.append(seq_len)
    return (np.asarray(data), np.asarray(labels), np.asarray(seq_lens))

if __name__ == "__main__":
    path = "data_dev/train/"
    dataset = load_dataset(path)
    batched_data, batched_labels, batched_seq_lens = make_batches(dataset)

