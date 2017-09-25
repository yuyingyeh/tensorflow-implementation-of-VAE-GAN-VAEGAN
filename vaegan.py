'''TensorFlow implementation of http://arxiv.org/pdf/1312.6114v10.pdf'''

from __future__ import absolute_import, division, print_function

import math

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope

from utils import encoder, decoder, discriminator
from generator import Generator


class VAEGAN(Generator):

    def __init__(self, hidden_size, batch_size, learning_rate, alpha, beta, gamma, sum_dir, attri_num, add_gan = 1, GAN_model = "V", similarity_layer = 4):
        
        print ("\nInitializing model with following parameters:")
        print ("batch_size:",batch_size, " learning_rate:", learning_rate, " alpha:", alpha, " beta:", beta, " gamma:", gamma)
        print ("GAN_model:", GAN_model, " similarity_layer:", similarity_layer, "\n")

        self.input_tensor = tf.placeholder(tf.float32, [batch_size, 64, 64 ,3])
        #self.input_label  = tf.placeholder(tf.int, [batch_size, attri_num])
        self.visual_attri = tf.placeholder(tf.float32, [hidden_size])

        with arg_scope([layers.conv2d, layers.conv2d_transpose],
                       activation_fn=tf.nn.relu,
                       normalizer_fn=layers.batch_norm,
                       normalizer_params={'scale': True},
                       padding='SAME'):
            with tf.variable_scope("model") as scope: #Full VAEGAN structure
                # Encoder
                ENC                  = encoder(self.input_tensor, hidden_size * 2)
                Enc_params_num       = len(tf.trainable_variables())

                # Add noise
                self.mean, stddev    = tf.split(1,2, ENC)
                stddev               = tf.sqrt(tf.exp(stddev))
                epsilon              = tf.random_normal([tf.shape(self.mean)[0], hidden_size])
                ENC_w_noise          = self.mean + epsilon * stddev

                # Decoder / Generator
                self.DEC_of_ENC      = decoder(ENC_w_noise)
                Enc_n_Dec_params_num = len(tf.trainable_variables())

                # Discriminator
                if add_gan == 1:
                    DIS_of_DEC_of_ENC    = discriminator(self.DEC_of_ENC, GAN_model)
                    Gen_dis_sum          = tf.scalar_summary('Gen_dis_mean', tf.reduce_mean(DIS_of_DEC_of_ENC))

            with tf.variable_scope("model", reuse=True) as scope: #Computation for Recon_Loss
                if add_gan == 1:
                    Real_Similarity      = discriminator(self.input_tensor, GAN_model, extract = similarity_layer)
                    Gen_Similarity       = discriminator(self.DEC_of_ENC, GAN_model, extract = similarity_layer) #+ tf.random_normal([batch_size, 8, 8, 256])

            with tf.variable_scope("model", reuse=True) as scope: #Computation for GAN_Loss
                if add_gan == 1:
                    Real_in_Dis          = discriminator(self.input_tensor, GAN_model)
                    Real_dis_sum         = tf.scalar_summary('Real_dis_mean', tf.reduce_mean(Real_in_Dis))
                    Prior_in_Dis         = discriminator(decoder(tf.random_normal([batch_size, hidden_size])), GAN_model)
                    Prior_dis_sum        = tf.scalar_summary('Prior_dis_mean', tf.reduce_mean(Prior_in_Dis))

            with tf.variable_scope("model", reuse=True) as scope: #Sample from latent space
                self.sampled_tensor  = decoder(tf.random_normal([batch_size, hidden_size]))

            with tf.variable_scope("model", reuse=True) as scope: #Add visual attributes
                #expand_mean = tf.expand_dims(self.mean, -1)
                print ("shape of mean:", np.shape(self.mean), " shape of visual attri:", np.shape(self.visual_attri))
                add_attri = self.mean + np.ones([batch_size, 1]) * self.visual_attri #[batch size, hidden size] (broadcasting)
                print ("shape of add attri:", tf.shape(add_attri))
                self.with_attri_tensor = decoder(add_attri)

        self.params     = tf.trainable_variables()
        self.Enc_params = self.params[:Enc_params_num]
        '''
        print ('Encoder Param:')
        for var in Enc_params:
            print (var.name)
        '''
        self.Dec_params = self.params[Enc_params_num:Enc_n_Dec_params_num]
        '''
        print ('Decoder Param:')
        for var in Dec_params:
            print (var.name)
        '''
        if add_gan == 1:
            self.Dis_params = self.params[Enc_n_Dec_params_num:]
        '''
        print ('Discriminator Param:')
        for var in Dis_params:
            print (var.name)
        '''
        self.Prior_loss = self.__get_prior_loss(self.mean, stddev)
        Prior_loss_sum  = tf.scalar_summary('Prior_loss', self.Prior_loss)
        if add_gan == 1:
            self.Recon_loss = self.__get_reconstruction_loss(Gen_Similarity, Real_Similarity)
            Recon_loss_sum  = tf.scalar_summary('Recon_loss', self.Recon_loss)
            self.GAN_loss   = self.__get_GAN_loss(Real_in_Dis, Prior_in_Dis, DIS_of_DEC_of_ENC, GAN_model)
            GAN_loss_sum    = tf.scalar_summary('GAN_loss', self.GAN_loss)
        else:
            self.Recon_loss = self.__get_reconstruction_loss(self.DEC_of_ENC, self.input_tensor)
            Recon_loss_sum  = tf.scalar_summary('Recon_loss', self.Recon_loss)

        # merge  summary for Tensorboard
        if add_gan == 1:
            self.detached_loss_summary_merged          =  tf.merge_summary([Prior_loss_sum,Recon_loss_sum,GAN_loss_sum,Real_dis_sum,Prior_dis_sum,Gen_dis_sum])
            #self.dis_mean_value_summary_merged         =  tf.merge_summary([Real_dis_sum,Prior_dis_sum,Gen_dis_sum])
        else:
            self.detached_loss_summary_merged          =  tf.merge_summary([Prior_loss_sum,Recon_loss_sum])

        if add_gan == 1:
            enc_loss   = self.Prior_loss + beta * self.Recon_loss
            dec_loss   = gamma * self.Recon_loss + self.GAN_loss
            dis_loss   = (-1) * self.GAN_loss
        else:
            total_loss = self.Prior_loss + beta * self.Recon_loss

        #self.combined_loss_summary_merged          =  tf.merge_summary([self.prior_loss_sum,self.recon_loss_sum,self.GAN_loss_sum])
        if add_gan == 1:
            self.train_enc = layers.optimize_loss(enc_loss, tf.contrib.framework.get_or_create_global_step(\
                ), learning_rate=learning_rate, variables = self.Enc_params, optimizer='RMSProp', update_ops=[])

            self.train_dec = layers.optimize_loss(dec_loss, tf.contrib.framework.get_or_create_global_step(\
                ), learning_rate=learning_rate, variables = self.Dec_params, optimizer='RMSProp', update_ops=[])

            self.train_dis = layers.optimize_loss(dis_loss, tf.contrib.framework.get_or_create_global_step(\
                ), learning_rate=learning_rate * alpha, variables = self.Dis_params, optimizer='RMSProp', update_ops=[])
        else:
            self.train     = layers.optimize_loss(total_loss, tf.contrib.framework.get_or_create_global_step(\
                ), learning_rate=learning_rate, variables = self.params, optimizer='RMSProp', update_ops=[])

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.initialize_all_variables())

        self.train_writer =  tf.train.SummaryWriter(sum_dir + '/train',self.sess.graph)

    def __get_prior_loss(self, mean, stddev, epsilon=1e-8):
        '''VAE loss
            KL-divergence, to restrict stddev not to decreasing to zero

        Args:
            mean:
            stddev:
            epsilon:
        '''
        return tf.reduce_mean(0.5 * (tf.square(mean) + tf.square(stddev) -
                                    2.0 * tf.log(stddev + epsilon) - 1.0))

    def __get_reconstruction_loss(self, output_tensor, target_tensor, epsilon=1e-8):
        '''Reconstruction loss

        Cross entropy reconstruction loss

        Args:
            output_tensor: tensor produces by decoder
            target_tensor: the target tensor that we want to reconstruct
            epsilon:
        '''
        #return tf.reduce_mean(-target_tensor * tf.log(output_tensor + epsilon) -
        #                     (1.0 - target_tensor) * tf.log(1.0 - output_tensor + epsilon))
        return tf.reduce_mean(tf.square(tf.sub(output_tensor, target_tensor)))

    def __get_cross_entropy_loss(self, output_tensor, target_tensor, epsilon=1e-8):
        '''Reconstruction loss

        Cross entropy reconstruction loss

        Args:
            output_tensor: tensor produces by decoder
            target_tensor: the target tensor that we want to reconstruct
            epsilon:
        '''
        output_tensor = np.add(np.multiply(output_tensor, 0.5), 0.5)
        target_tensor = np.add(np.multiply(target_tensor, 0.5), 0.5)
        return tf.reduce_mean(-target_tensor * tf.log(output_tensor + epsilon) -
                             (1.0 - target_tensor) * tf.log(1.0 - output_tensor + epsilon))

    def __get_GAN_loss(self, dis_of_real, dis_of_dec_of_z, dis_result, gan_model, epsilon=1e-8):
        '''GAN loss

        GAN loss 

        Args:
            output_tensor: tensor produces by decoder
            target_tensor: the target tensor that we want to reconstruct
            epsilon:
        '''
        if gan_model == "LS":
            return -tf.reduce_mean(tf.square(dis_of_real ) + tf.square(1.0 - dis_of_dec_of_z )
                             + tf.square(1.0 - dis_result ))
        else:
            return tf.reduce_mean(tf.log(dis_of_real + epsilon) + tf.log(1.0 - dis_of_dec_of_z + epsilon)
                             + tf.log(1.0 - dis_result + epsilon))
        

    def update_params(self, input_tensor, add_gan = 1):
        '''Update parameters of the network

        Args:
            input_tensor: a batch of flattened images [batch_size, 28*28]

        Returns:
            Current loss value
        '''
        '''
        loss_of_Enc = self.sess.run(self.train_enc, {self.input_tensor: input_tensor})
        loss_of_Dec = self.sess.run(self.train_dec, {self.input_tensor: input_tensor})
        loss_of_Dis = self.sess.run(self.train_dis, {self.input_tensor: input_tensor})
        '''
        if add_gan == 1:

            loss_of_Enc, loss_of_Dec, loss_of_Dis, loss_summary = self.sess.run([self.train_enc, self.train_dec, \
                self.train_dis, self.detached_loss_summary_merged], {self.input_tensor: input_tensor})
            return loss_of_Enc, loss_of_Dec, loss_of_Dis, loss_summary
        else:
            loss, loss_summary = self.sess.run([self.train, self.detached_loss_summary_merged], \
                {self.input_tensor: input_tensor})
            return loss, loss_summary

    def cal_attri_vector(self, input_tensor, attri_label):
        latent_rep = self.sess.run(self.mean, {self.input_tensor: input_tensor})

        batch_size = tf.shape(attri_label)[0]
        pos_attri_label = (attri_label + 1)/2
        neg_attri_label = (attri_label - 1)/(-2)
        #print (pos_attri_label)
        pos_label_sum = tf.reduce_sum(pos_attri_label, 0) 
        print (pos_label_sum)
        neg_label_sum = tf.reduce_sum(neg_attri_label, 0)
        #pos_label_sum = tf.ones([batch_size, 1], tf.float32) * pos_label_sum
        #neg_label_sum = tf.ones([batch_size, 1], tf.float32) * neg_label_sum
        #pos_attri_label = tf.div(pos_attri_label, pos_label_sum)
        #neg_attri_label = tf.div(neg_attri_label, neg_label_sum)
        pos_attri_sum = tf.transpose(latent_rep) * pos_attri_label #[hidden size, label num]
        neg_attri_sum = tf.transpose(latent_rep) * neg_attri_label #[hidden size, label num]
                    
        return pos_attri_sum, pos_label_sum, neg_attri_sum, neg_label_sum
            
        '''
    def add_attri(self, input_tensor, attri_vector):
            with_attri_imgs = self.sess.run([self.with_attri_tensor], {self.input_tensor: input_tensor, self.visual_attri: attri_vector})
        return with_attri_imgs
        '''