import time
import six
import sys

from imagenet_data import ImagenetData
import image_processing
import model_zoo
import numpy as np
import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'imagenet', 'Imagenet classification')
tf.app.flags.DEFINE_string('subset', 'train', 'Imagenet training dataset')
tf.app.flags.DEFINE_integer('num_classes', 1001, 'Number classes.')
tf.app.flags.DEFINE_integer('epoch', 100, 'Number of epoch.')
tf.app.flags.DEFINE_integer('num_steps', 1000001, 'Number of step.')
tf.app.flags.DEFINE_float('power', 0.9, 'power of lr.')
tf.app.flags.DEFINE_float('weight_decay', 0.0001, 'decay.')
tf.app.flags.DEFINE_integer('top_k', 5, 'top k accuracy.')
tf.app.flags.DEFINE_string('SNAPSHOT_DIR', './train',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_integer('gpu_nums', 4,
                            'Number of gpus used for training')
tf.app.flags.DEFINE_float('learning_rate', 0.1, 'lr.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 20.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('MOVING_AVERAGE_DECAY', 0.99, 'MOVING_AVERAGE_DECAY = 0.99.')

RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

class Model_Graph(object):
    def __init__(self, num_class = 1001, is_training = True):
        self.num_class = num_class
        self.is_training = is_training

    def _build_defaut_graph(self, images):
        """
        Densenet
        """
        model = model_zoo.Densenet(num_class = self.num_class,
                                     images = images, is_training = self.is_training)
        model.build_graph()

        return model

def save(saver, sess, logdir, step):
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)

   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def main(_):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        dataset = ImagenetData(subset=FLAGS.subset)
        assert dataset.data_files()
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        # Calculate the learning rate schedule.
        num_batches_per_epoch = (dataset.num_examples_per_epoch() /FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps.
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)

        tf.summary.scalar('lr', learning_rate)

        is_training = tf.placeholder(tf.bool)

        #opt = tf.train.AdamOptimizer(learning_rate)
        opt = tf.train.RMSPropOptimizer(learning_rate, RMSPROP_DECAY,
                                        momentum=RMSPROP_MOMENTUM,
                                        epsilon=RMSPROP_EPSILON)

        with tf.name_scope("create_inputs"):
            #if tf.gfile.Exists(FLAGS.SNAPSHOT_DIR):
            #    tf.gfile.DeleteRecursively(FLAGS.SNAPSHOT_DIR)
            #tf.gfile.MakeDirs(FLAGS.SNAPSHOT_DIR)

            # Get images and labels for ImageNet and split the batch across GPUs.
            assert FLAGS.batch_size % FLAGS.gpu_nums == 0, ('Batch size must be divisible by number of GPUs')
            split_batch_size = int(FLAGS.batch_size / FLAGS.gpu_nums)

            # Override the number of preprocessing threads to account for the increased
            # number of GPU towers.
            num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.gpu_nums
            images, labels = image_processing.distorted_inputs(dataset, num_preprocess_threads=num_preprocess_threads)
            #tf.summary.image('images', images, max_outputs = 10)

            images_splits = tf.split(axis=0, num_or_size_splits=FLAGS.gpu_nums, value=images)
            labels_splits = tf.split(axis=0, num_or_size_splits=FLAGS.gpu_nums, value=tf.one_hot(indices = labels, depth = FLAGS.num_classes))

        multi_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.gpu_nums):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('ImageNet', i)) as scope:

                        graph = Model_Graph(num_class = FLAGS.num_classes, is_training = is_training)

                        model = graph._build_defaut_graph(images = images_splits[i])

                        # Top-1 accuracy
                        top1acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(model.logits, tf.argmax(labels_splits[i], axis=1), 1), tf.float32))
                        # Top-n accuracy
                        topnacc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(model.logits, tf.argmax(labels_splits[i], axis=1), FLAGS.top_k), tf.float32))

                        tf.summary.scalar('top1acc_{}'.format(i), top1acc)
                        tf.summary.scalar('topkacc_{}'.format(i), topnacc)

                        all_trainable = [v for v in tf.trainable_variables()]

                        loss = tf.nn.softmax_cross_entropy_with_logits(logits=model.logits, labels=labels_splits[i])

                        l2_losses = [FLAGS.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
                        reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)

                        tf.summary.scalar('loss_{}'.format(i), reduced_loss)

                        tf.get_variable_scope().reuse_variables()

                        #batchnorm_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
                        batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                        grads = opt.compute_gradients(reduced_loss, all_trainable)
                        multi_grads.append(grads)

        grads = average_gradients(multi_grads)

        # Track the moving averages of all trainable variables.
        # Note that we maintain a "double-average" of the BatchNormalization
        # global statistics. This is more complicated then need be but we employ
        # this for backward-compatibility with our previous models.
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.MOVING_AVERAGE_DECAY, global_step)

        variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)

        # Group all updates to into a single train op.
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        train_op = tf.group(opt.apply_gradients(grads, global_step), variables_averages_op, batchnorm_updates_op)

        #grads_value = list(zip(grads, all_trainable))
        #for grad, var in grads_value:
        #    tf.summary.histogram(var.name + '/gradient', grad)

        summary_op = tf.summary.merge_all()

        # Set up tf session and initialize variables. 
        config = tf.ConfigProto()
        config.allow_soft_placement=True
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()

        sess.run(init)

        # Saver for storing checkpoints of the model.
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=2)

        restore_var = [v for v in tf.trainable_variables()]+[v for v in tf.global_variables() if 'moving_mean' in v.name or 'moving_variance' in v.name or 'global_step' in v.name]

        ckpt = tf.train.get_checkpoint_state(FLAGS.SNAPSHOT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            loader = tf.train.Saver(var_list=restore_var)
            load(loader, sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found.')
            load_step = 0


        summary_writer = tf.summary.FileWriter(FLAGS.SNAPSHOT_DIR, graph=sess.graph)

        # Iterate over training steps.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        for step in range(FLAGS.num_steps):
            start_time = time.time()

            feed_dict = {is_training: True}
            if step%50000 == 0 and step != 0:
                loss_value, _ = sess.run([reduced_loss, train_op], feed_dict=feed_dict)
                save(saver, sess, FLAGS.SNAPSHOT_DIR, step)
            elif step%100 == 0:
                summary_str, loss_value, _ = sess.run([summary_op, reduced_loss, train_op], feed_dict=feed_dict)
                duration = time.time() - start_time
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
                print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
            else:
                loss_value, _ = sess.run([reduced_loss, train_op], feed_dict=feed_dict)

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)                                                                                  
