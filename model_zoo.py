import numpy as np
import tensorflow as tf
import six
slim = tf.contrib.slim

class Densenet(object):
    """Densenet model."""
    def __init__(self, num_class, images, is_training):
        self.num_classes = num_class
        self._images = images
        self.training  = is_training
    
    def build_graph(self):
        """Build a whole graph for the model."""
        self._build_model()
    
    def _build_model(self):
        with slim.arg_scope(self.densenet_arg_scope()):
            self.logits, _ = self.densenet121(self._images, num_classes=self.num_classes, is_training=self.training)
            #self.logits, _ = self.densenet161(self._images, num_classes=self.num_classes, is_training=self.training)
            #self.logits, _ = self.densenet169(self._images, num_classes=self.num_classes, is_training=self.training)

    @slim.add_arg_scope
    def _global_avg_pool2d(self, inputs, data_format='NHWC', scope=None, outputs_collections=None):
        with tf.variable_scope(scope, 'xx', [inputs]) as sc:
            axis = [1, 2] if data_format == 'NHWC' else [2, 3]
            net = tf.reduce_mean(inputs, axis=axis, keep_dims=True)
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
            return net


    @slim.add_arg_scope
    def _conv(self, inputs, num_filters, kernel_size, stride=1, dropout_rate=None, scope=None, outputs_collections=None):
        with tf.variable_scope(scope, 'xx', [inputs]) as sc:
            net = slim.batch_norm(inputs)
            net = tf.nn.relu(net)
            net = slim.conv2d(net, num_filters, kernel_size)

            if dropout_rate:
                net = tf.nn.dropout(net)

            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

        return net


    @slim.add_arg_scope
    def _conv_block(self, inputs, num_filters, data_format='NHWC', scope=None, outputs_collections=None):
        with tf.variable_scope(scope, 'conv_blockx', [inputs]) as sc:
            net = inputs
            net = self._conv(net, num_filters*4, 1, scope='x1')
            net = self._conv(net, num_filters, 3, scope='x2')
            if data_format == 'NHWC':
                net = tf.concat([inputs, net], axis=3)
            else: # "NCHW"
                net = tf.concat([inputs, net], axis=1)

            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

        return net


    @slim.add_arg_scope
    def _dense_block(self, inputs, num_layers, num_filters, growth_rate,
                 grow_num_filters=True, scope=None, outputs_collections=None):

        with tf.variable_scope(scope, 'dense_blockx', [inputs]) as sc:
            net = inputs
            for i in range(num_layers):
                branch = i + 1
                net = self._conv_block(net, growth_rate, scope='conv_block'+str(branch))

                if grow_num_filters:
                    num_filters += growth_rate

            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

        return net, num_filters


    @slim.add_arg_scope
    def _transition_block(self, inputs, num_filters, compression=1.0,
                      scope=None, outputs_collections=None):

        num_filters = int(num_filters * compression)
        with tf.variable_scope(scope, 'transition_blockx', [inputs]) as sc:
            net = inputs
            net = self._conv(net, num_filters, 1, scope='blk')

            net = slim.avg_pool2d(net, 2)

            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

        return net, num_filters


    def densenet(self, inputs,
             num_classes=1000,
             reduction=None,
             growth_rate=None,
             num_filters=None,
             num_layers=None,
             dropout_rate=None,
             data_format='NHWC',
             is_training=True,
             reuse=None,
             scope=None):
        assert reduction is not None
        assert growth_rate is not None
        assert num_filters is not None
        assert num_layers is not None

        compression = 1.0 - reduction
        num_dense_blocks = len(num_layers)

        if data_format == 'NCHW':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        with tf.variable_scope(scope, 'densenetxxx', [inputs, num_classes],
                         reuse=reuse) as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                             is_training=is_training), \
                slim.arg_scope([slim.conv2d, self._conv, self._conv_block,
                             self._dense_block, self._transition_block], 
                             outputs_collections=end_points_collection), \
                slim.arg_scope([self._conv], dropout_rate=dropout_rate):
                net = inputs

                # initial convolution
                net = slim.conv2d(net, num_filters, 7, stride=2, scope='conv1')
                net = slim.batch_norm(net)
                net = tf.nn.relu(net)
                net = slim.max_pool2d(net, 3, stride=2, padding='SAME')

                # blocks
                for i in range(num_dense_blocks - 1):
                    # dense blocks
                    net, num_filters = self._dense_block(net, num_layers[i], num_filters,
                                        growth_rate,
                                        scope='dense_block' + str(i+1))

                    # Add transition_block
                    net, num_filters = self._transition_block(net, num_filters,
                                             compression=compression,
                                             scope='transition_block' + str(i+1))

                net, num_filters = self._dense_block(
                    net, num_layers[-1], num_filters,
                    growth_rate,
                    scope='dense_block' + str(num_dense_blocks))

                # final blocks
                with tf.variable_scope('final_block', [inputs]):
                    net = slim.batch_norm(net)
                    net = tf.nn.relu(net)
                    net = self._global_avg_pool2d(net, scope='global_avg_pool')

                net = slim.conv2d(net, num_classes, 1,
                        biases_initializer=tf.zeros_initializer(),
                        scope='logits')

                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                if num_classes is not None:
                    end_points['predictions'] = slim.softmax(net, scope='predictions')

                return tf.reduce_mean(net,[1,2]), end_points


    def densenet121(self, inputs, num_classes=1000, data_format='NHWC', is_training=True, reuse=None):
        return self.densenet(inputs,
                  num_classes=num_classes, 
                  reduction=0.5,
                  growth_rate=32,
                  num_filters=64,
                  num_layers=[6,12,24,16],
                  data_format=data_format,
                  is_training=is_training,
                  reuse=reuse,
                  scope='densenet121')
    densenet121.default_image_size = 224


    def densenet161(self, inputs, num_classes=1000, data_format='NHWC', is_training=True, reuse=None):
        return self.densenet(inputs,
                  num_classes=num_classes, 
                  reduction=0.5,
                  growth_rate=48,
                  num_filters=96,
                  num_layers=[6,12,36,24],
                  data_format=data_format,
                  is_training=is_training,
                  reuse=reuse,
                  scope='densenet161')
    densenet161.default_image_size = 224


    def densenet169(self, inputs, num_classes=1000, data_format='NHWC', is_training=True, reuse=None):
        return self.densenet(inputs,
                  num_classes=num_classes, 
                  reduction=0.5,
                  growth_rate=32,
                  num_filters=64,
                  num_layers=[6,12,32,32],
                  data_format=data_format,
                  is_training=is_training,
                  reuse=reuse,
                  scope='densenet169')
    densenet169.default_image_size = 224


    def densenet_arg_scope(self, weight_decay=1e-4,
                       batch_norm_decay=0.99,
                       batch_norm_epsilon=1.1e-5,
                       data_format='NHWC'):
        with slim.arg_scope([slim.conv2d, slim.batch_norm, slim.avg_pool2d, slim.max_pool2d,
                       self._conv_block, self._global_avg_pool2d],
                      data_format=data_format):
            with slim.arg_scope([slim.conv2d],
                         weights_regularizer=slim.l2_regularizer(weight_decay),
                         activation_fn=None,
                         biases_initializer=None):
                with slim.arg_scope([slim.batch_norm],
                          scale=True,
                          decay=batch_norm_decay,
                          epsilon=batch_norm_epsilon) as scope:
                    return scope
