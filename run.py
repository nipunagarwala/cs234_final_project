# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import models
import utils
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys


GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.per_process_gpu_memory_fraction = 0.5


def run_model(args):
    dataset_dir = utils.choose_data(args)
    print "Using checkpoint directory: {0}".format(args.ckpt_dir)

    model = utils.choose_model(args) # pass in necessary model parameters
    print "Running {0} model for {1} epochs.".format(args.model, args.num_epochs)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    saver = tf.train.Saver(max_to_keep=NUM_EPOCHS)

    with tf.Session(config=GPU_CONFIG) as session:
        print "Inititialized TF Session!"

        # Checkpoint
        i_stopped, found_ckpt = utils.get_checkpoint(args, session, saver)

        # Summary Writer
        file_writer = tf.summary.FileWriter(args.ckpt_dir, graph=session.graph, max_queue=10, flush_secs=30)

		# Val or Test set accuracies
		batch_accuracies = []

        # Make computational graph
        if args.train == "train":
            init_op = tf.global_variables_initializer()
            init_op.run()
        else:
            if not found_ckpt:
                print "No checkpoint found for test or validation!"
                return

        for i in xrange(i_stopped, args.num_epochs):
            print "Running epoch ({0})...".format(i)
				# Shuffle dataset on new epoch
                random.shuffle(dateset)
                batched_data, batched_labels, batched_seq_lens = utils.make_batches(dataset, n=batch_size)
                for j in xrange(len(data_batches)):
                    data_batch = batched_data[j]
					label_batch = batched_labels[j]
					seq_lens_batch = batched_seq_lens[j]

                    # pack feed dict
					# feed_dict = None

                    summary, accuracy = model.run(args, session, feed_dict)
                    file_writer.add_summary(summary, j)

                    # Record batch accuracies for test code
                    if args.train == "test" or args.train == 'val':
                        batch_accuracies.append(accuracy)

            if args.train == "train":
                # Checkpoint model - every epoch
                utils_runtime.save_checkpoint(args, session, saver, i)
            else: # val or test
                test_accuracy = np.mean(batch_accuracies)
                print "Model {0} accuracy: {1}".format(args.train, test_accuracy)


def main(_):
    args = utils.parse_command_line()
    # utils.clear_summaries()
    run_model(args)


if __name__ == "__main__":
    tf.app.run()
