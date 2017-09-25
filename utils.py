import tensorflow as tf
from tensorflow.contrib import layers


def encoder(input_tensor, output_size, extract = -1):
    '''Create encoder network.

    Args:
        input_tensor: a batch of flattened images [batch_size, 28*28]

    Returns:
        A tensor that expresses the encoder network
    '''
    #net = tf.reshape(input_tensor, [-1, 64, 64, 3])
    net = layers.conv2d(input_tensor, 64, 5, stride=2, scope="Enc_conv_1") #[32,32,64]
    net = layers.conv2d(net, 128, 5, stride=2, scope="Enc_conv_2") #[16,16,128]
    net = layers.conv2d(net, 256, 5, stride=2, scope="Enc_conv_3") #[8,8,256]

    if extract == -1:
        #net = layers.dropout(net, keep_prob=0.9)
        net = layers.flatten(net)
        net = layers.fully_connected(net, 1024, activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm, normalizer_params={'scale': True}, scope="Enc_fc_1")
        net = layers.fully_connected(net, output_size, activation_fn=None, scope="Enc_fc_2")
        return net
    else:
        return net


def decoder(input_tensor):
    '''Create decoder network.

        If input tensor is provided then decodes it, otherwise samples from
        a sampled vector.
    Args:
        input_tensor: a batch of vectors to decode

    Returns:
        A tensor that expresses the decoder network
    '''
    dim = 8*8*256
    net = layers.fully_connected(input_tensor, dim, activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm, normalizer_params={'scale': True}, scope="Dec_fc_1")
    #net = tf.expand_dims(net, 1)
    #net = tf.expand_dims(net, 1)
    net = tf.reshape(net, [-1, 8, 8, 256])
    net = layers.conv2d_transpose(net, 256, 5, stride=2, scope="Dec_conv_tran_1") #[16,16,256]
    net = layers.conv2d_transpose(net, 128, 5, stride=2, scope="Dec_conv_tran_2") #[32,32,128]
    net = layers.conv2d_transpose(net, 32, 5, stride=2, scope="Dec_conv_tran_3") #[64,64,32]
    net = layers.conv2d_transpose(net, 3, 5, stride=1, activation_fn=tf.tanh, normalizer_fn=None, scope="Dec_conv_tran_4") #[64,64,3]
    #net = layers.flatten(net)
    return net


def discriminator(input_tensor, gan_model, extract = -1):
    '''Create a network that discriminates between images from a dataset and
    generated ones.

    Args:
        input: a batch of real images [batch, height, width, channels]
    Returns:
        A tensor that represents the network
    '''
    #net = tf.reshape(input_tensor, [-1, 64, 64, 3])
    if extract == 0:
        return input_tensor

    net = layers.conv2d(input_tensor, 32, 5, stride=1, scope="Dis_conv_1", normalizer_fn=None) #[64,64,32]
    if extract == 1:
        return net

    net = layers.conv2d(net, 128, 5, stride=2, scope="Dis_conv_2") #[32,32,128]
    if extract == 2:
        return net

    net = layers.conv2d(net, 256, 5, stride=2, scope="Dis_conv_3") #[16,16,256]
    if extract == 3:
        return net

    net = layers.conv2d(net, 256, 5, stride=2, scope="Dis_conv_4") #[8,8,256]
    if extract == 4:
        return net

    #net = layers.dropout(net, keep_prob=0.9)
    net = layers.flatten(net)
    net = layers.fully_connected(net, 512, activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm, normalizer_params={'scale': True}, scope="Dis_fc_1")
    if gan_model == "LS":
        net = layers.fully_connected(net, 1, activation_fn=None, scope="Dis_fc_2")
    else:
        net = layers.fully_connected(net, 1, activation_fn=tf.sigmoid, scope="Dis_fc_2")
    return net



