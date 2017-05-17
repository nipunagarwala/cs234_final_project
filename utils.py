import numpy as np 
import sys
import os
import argparse



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

'''
def make_batches():


'''
    Reads in the videos and corresponding labels into memory and
    returns them respectively.

'''
def load_dataset():




