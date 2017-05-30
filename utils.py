import numpy as np
import sys
import os
from argparse import ArgumentParser
import cPickle as pickle
import re

from RecurrentCNN import *
from VisualAttention import *

# TODO: change according to data directories
# TRAIN_DATA = '/data/MOT17/data/train/'
# TRAIN_DATA = '/data/vot2017/data/train/'
# TEST_DATA = '/data/vot2017/data/test/'
TRAIN_DATA = '/data/newvot2017/data/train/'
TEST_DATA = '/data/newvot2017/data/test/'
# VALIDATION_DATA = '/data/validation_data/'
SUMMARY_DIR = '/data/summary'


################################# Logging ###################################

def clear_summaries():
    if tf.gfile.Exists(SUMMARY_DIR):
        tf.gfile.DeleteRecursively(SUMMARY_DIR)
    tf.gfile.MakeDirs(SUMMARY_DIR)


def get_checkpoint(args, session, saver):
	# Checkpoint
	found_ckpt = False

	if args.override:
		if tf.gfile.Exists(args.ckpt_dir):
			tf.gfile.DeleteRecursively(args.ckpt_dir)
		tf.gfile.MakeDirs(args.ckpt_dir)

	# check if args.ckpt_dir is a directory of checkpoints, or the checkpoint itself
	if len(re.findall('model.ckpt-[0-9]+', args.ckpt_dir)) == 0:
		ckpt = tf.train.get_checkpoint_state(args.ckpt_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(session, ckpt.model_checkpoint_path)
			i_stopped = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
			print "Found checkpoint for epoch ({0})".format(i_stopped)
			found_ckpt = True
		else:
			print('No checkpoint file found!')
			i_stopped = 0
	else:
		saver.restore(session, args.ckpt_dir)
		i_stopped = int(args.ckpt_dir.split('/')[-1].split('-')[-1])
		print "Found checkpoint for epoch ({0})".format(i_stopped)
		found_ckpt = True


	return i_stopped, found_ckpt


def save_checkpoint(args, session, saver, i):
	checkpoint_path = os.path.join(args.ckpt_dir, 'model.ckpt')
	saver.save(session, checkpoint_path, global_step=i)
	# saver.save(session, os.path.join(SUMMARY_DIR,'model.ckpt'), global_step=i)


################################ Command Line #################################


def parse_command_line():
    desc = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)
    parser = ArgumentParser(description=desc)

    print("Parsing Command Line Arguments...")
    requiredModel = parser.add_argument_group('Required Model arguments')
    # TODO: add other model names
    requiredModel.add_argument('-m', choices = ["rnn_rcnn", "rnn_rcnn_cumsum", "visual_attention"], type=str,
                        dest='model', required=True, help='Type of model to run')
    requiredTrain = parser.add_argument_group('Required Train/Test arguments')
    requiredTrain.add_argument('-p', choices = ["train", "val", "test"], type=str, # inference mode?
    					dest='train', required=True, help='Training or Testing phase to be run')

    parser.add_argument('-data', dest='data_dir', default=TRAIN_DATA, help='Specify the train data directory')
    parser.add_argument('-o', dest='override', action="store_true", help='Override the checkpoints')
    parser.add_argument('-e', dest='num_epochs', default=50, type=int, help='Set the number of Epochs')
    parser.add_argument('-ckpt', dest='ckpt_dir', default='/data/ckpts/temp_ckpt/', type=str, help='Set the checkpoint directory')

    args = parser.parse_args()
    return args


def choose_data(args):
    if args.data_dir != '':
        dataset_dir = args.data_dir
    elif args.train == 'train':
        dataset_dir = TRAIN_DATA
    elif args.train == 'test':
        dataset_dir = TEST_DATA
    else: # args.train == 'dev'
        dataset_dir = VALIDATION_DATA

    print 'Using dataset {0}'.format(dataset_dir)
    print "Reading in {0}-set filenames.".format(args.train)
    return dataset_dir


def choose_model(args): # pass in necessary model parameters (...)
    is_training = args.train == 'train' # boolean that certain models may require

    if args.model == 'rnn_rcnn':
        # features_shape = (240, 384, 3) # MOT17
        features_shape = (180, 320, 3) # vot2017
        num_classes = 4
        seq_len = 8

        model = RecurrentCNN(features_shape,
                        num_classes,
                        cell_type='lstm',
                        seq_len=seq_len,
                        reuse=False,
                        add_bn=False,
                        add_reg=False,
                        deeper=True,
                        scope="rnn_rcnn")
        model.build_model()
        model.add_loss_op()
        model.add_error_op()
        model.add_optimizer_op()
        model.add_summary_op()
    elif args.model == 'rnn_rcnn_cumsum':
        features_shape = (180, 320, 3) # vot2017
        num_classes = 4
        seq_len = 8

        model = RecurrentCNN(features_shape,
                        num_classes,
                        cell_type='lstm',
                        seq_len=seq_len,
                        reuse=False,
                        add_bn=False,
                        add_reg=False,
                        deeper = True,
                        scope="rnn_rcnn_cumsum")
        model.build_model()
        model.add_cumsum_loss_op()
        model.add_error_op()
        model.add_optimizer_op()
        model.add_summary_op()
    elif args.model == 'visual_attention':
        pass
    elif args.model == 'other':
        pass

    return model

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
    batched_bbox = []
    num_batches = int(np.ceil(len(data) / float(batch_size)))

    for i in range(num_batches):
        batch_start = i * batch_size
        batched_data.append(np.asarray(data[batch_start : batch_start + batch_size]))
        batched_labels.append(np.asarray(labels[batch_start : batch_start + batch_size]))
        batched_seq_lens.append(np.asarray(seq_lens[batch_start : batch_start + batch_size]))
        bboxes = [seq[0][0:4] for seq in labels[batch_start : batch_start + batch_size]]
        batched_bbox.append(np.asarray(bboxes))
    return batched_data, batched_labels, batched_seq_lens, batched_bbox


'''
    Reads in the videos and corresponding labels into memory and
    returns them respectively.

'''
def load_dataset(dataset_path):
    data = []
    labels = []
    seq_lens = []
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        for filename in filenames:
            with open(os.path.join(dirpath, filename), 'rb') as f:
                curr_seq, curr_labels, seq_len = pickle.load(f)
                data.append(np.asarray(curr_seq))
                labels.append(np.asarray(curr_labels))
                #print curr_labels
                #print np.asarray(labels).shape
                seq_lens.append(np.asarray(seq_len))
    return (np.asarray(data), np.asarray(labels), np.asarray(seq_lens))


if __name__ == "__main__":
    path = "/data/MOT17/data/train"
    dataset = load_dataset(path)
    batched_data, batched_labels, batched_seq_lens, batched_bbox = make_batches(dataset)
