import tensorflow as tf

def conv2d(x,
           output_size,
           kernel_size,
           stride_size,
           initializer=tf.truncated_normal_initializer(0, 0.02),
           activation_fn=tf.nn.relu,
           padding='VALID',
           name='conv2d'):
    with tf.variable_scope(name):
        stride = [1, stride_size[0], stride_size[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_size]

        w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
        b = tf.get_variable('biases', [output_size], initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(x, w, stride, padding)

        output = tf.nn.bias_add(conv, b)

    if activation_fn != None:
        return activation_fn(output), w, b
    else:
        return output, w, b


def linear(x,
           output_size,
           stddev=0.02,
           bias_init=0.0,
           activation_fn=None,
           name='linear'):
    shape = x.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable('w', [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_size], initializer=tf.constant_initializer(bias_init))

        out = tf.nn.bias_add(tf.matmul(x,w), b)

    if activation_fn != None:
        return activation_fn(out), w, b
    else:
        return out, w, b



