'''TensorFlow implementation of http://arxiv.org/pdf/1312.6114v10.pdf'''

from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np
import scipy.misc
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope
from scipy.misc import imsave

from vae import VAE
from gan import GAN
from vaegan import VAEGAN

import h5py
from random import shuffle
import time

#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5) 
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_integer("updates_per_epoch", 250, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_float("learning_rate", 3e-4, "learning rate")
flags.DEFINE_string("image_directory", "results/", "image_directory")
flags.DEFINE_string("summary_directory", "summary/", "summary_directory")
flags.DEFINE_string("model_directory", "model/", "model_directory")
flags.DEFINE_integer("hidden_size", 128, "size of the hidden VAE unit")

flags.DEFINE_string("mode", "train", "train or test")
flags.DEFINE_string("model", "vaegan", "vaegan or vae or gan")
flags.DEFINE_string("GAN_model", "V", "")
flags.DEFINE_integer("similarity_layer", 4, "parameter for vaegan only")

flags.DEFINE_string("beta", 20, "coef of recon_loss in ENC")
flags.DEFINE_string("alpha", 0.1, "coef of learning rate of DIS")
flags.DEFINE_string("gamma", 0.5, "recon or fool discriminator")

flags.DEFINE_string("restore", False, "restore model")
flags.DEFINE_string("store", True, "store model")
flags.DEFINE_integer("save_model_freq", 5000, "")
flags.DEFINE_integer("save_img_freq", 1000, "")

FLAGS = flags.FLAGS

def shuffle_train_idx(length):
    x = [i for i in range(length)]
    shuffle(x)
    return x

def get_batch(dataset,batch_size,idx,k):
    batch = np.array([dataset[idx[i+batch_size*k]]  for i in range(batch_size)])
    return batch

def pre_process(input_npary):
    tmp=np.subtract(input_npary,128.)       
    tmp=np.divide(tmp,128.)
    return tmp

if __name__ == "__main__":

    assert FLAGS.model in ['vae', 'vaegan']
    if FLAGS.model == 'vae':
        add_gan = 0
        GAN_model = ''
        similarity_layer = 0
    elif FLAGS.model == 'vaegan':
        add_gan = 1
        GAN_model = FLAGS.GAN_model
        similarity_layer = FLAGS.similarity_layer

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_dir = FLAGS.summary_directory + FLAGS.model + '/bs-' +str(FLAGS.batch_size) + '|a-'+ str(FLAGS.alpha)+ '|b-'+ str(FLAGS.beta)+ '|g-'+ str(FLAGS.gamma) +'|sl-'+ str(similarity_layer) +'|'+ GAN_model +'|'+ start_time
    img_dir = FLAGS.image_directory + FLAGS.model + '/bs-' +str(FLAGS.batch_size) + '|a-'+ str(FLAGS.alpha)+ '|b-'+ str(FLAGS.beta)+ '|g-'+ str(FLAGS.gamma) +'|sl-'+ str(similarity_layer) +'|'+ GAN_model +'|'+ start_time
    model_dir = FLAGS.model_directory + FLAGS.model + '/bs-' +str(FLAGS.batch_size) + '|a-'+ str(FLAGS.alpha)+ '|b-'+ str(FLAGS.beta)+ '|g-'+ str(FLAGS.gamma)+'|sl-'+ str(similarity_layer) +'|'+ GAN_model 
    enc_model_dir = model_dir + '/Enc.ckpt'
    dec_model_dir = model_dir + '/Dec.ckpt'
    dis_model_dir = model_dir + '/Dis.ckpt'

    # Load Data
    data_path = 'dataset/CelebA_attri.h5'
    with h5py.File(data_path,'r') as data:
        img_train        =  data['/CelebA/train/img'][()]

    print('Training Data Loaded')
    img_train = np.transpose(img_train, (0, 2, 3, 1))

    # Iteration number and data number
    img_train_num              =  len(img_train)                # 166007
    img_train_iter_num         =  img_train_num / FLAGS.batch_size   #  39999

    # training VAE
    if FLAGS.model == "VAE":
        # initialize model
        model = VAEGAN(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.alpha, FLAGS.beta, FLAGS.gamma, log_dir, FLAGS.attri_num, \
        add_gan, GAN_model, similarity_layer)

        # restore the model(parameter )
        ENC_saver = tf.train.Saver(var_list=model.Enc_params)
        DEC_saver = tf.train.Saver(var_list=model.Dec_params)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if FLAGS.restore == True:
            ENC_saver.restore(model.sess, enc_model_dir)
            print("Encoder Model restored in file: %s" % enc_model_dir)
            DEC_saver.restore(model.sess, dec_model_dir)
            print("Decoder Model restored in file: %s" % dec_model_dir)
            
        # start training
        for epoch in range(FLAGS.max_epoch):
            # Shuffle Real Photo every epoch
            img_train_idx     =  shuffle_train_idx(img_train_num)

            for i in (range(int(img_train_iter_num))):
                # Get the batch
                batch_img   = get_batch(img_train, FLAGS.batch_size, img_train_idx , i)
                input_img   = pre_process(batch_img)

                # update model
                print ("\nEpoch: ", epoch, " / ", FLAGS.max_epoch, " ; Iter: ", i, " / ", int(img_train_iter_num))
                loss, loss_summary = model.update_params(input_img, add_gan)
                print ("Total Loss : %f " % loss)

                # write log to tensorboard
                model.train_writer.add_summary(loss_summary, i+epoch*img_train_iter_num)
                
                # Save weights
                if (i+epoch*int(img_train_iter_num))% FLAGS.save_model_freq == FLAGS.save_model_freq-1:
                    if FLAGS.store == True:
                        ENC_save_path = ENC_saver.save(model.sess, model_dir + '/Enc.ckpt')
                        print("Encoder Model saved in file: %s" % ENC_save_path)
                        DEC_save_path = DEC_saver.save(model.sess, model_dir + '/Dec.ckpt')
                        print("Decoder Model saved in file: %s" % DEC_save_path)

                # Save generated images
                if i%FLAGS.save_img_freq==0:
                    print ('\nsaving compact images')
                    #compact_training_img = model.compact_batch_img(batch_img[0:8],batch_img[8:16],batch_img[16:24])

                    imgs_folder = os.path.join(img_dir, 'imgs')
                    if not os.path.exists(imgs_folder):
                        os.makedirs(imgs_folder)
                    
                    imgnum = i/FLAGS.save_img_freq+img_train_iter_num/FLAGS.save_img_freq*epoch
                    #imsave(os.path.join(imgs_folder, 'training_img_iter%d.png') % imgnum, compact_training_img)
                    model.generate_and_save_compact_images(FLAGS.batch_size, input_img, img_dir, imgnum)

    # training GAN
    elif FLAGS.model == "GAN":

    # traiing VAEGAN
    elif FLAGS.model == "VAEGAN":

    # restore the model(parameter )
    ENC_saver = tf.train.Saver(var_list=model.Enc_params)
    DEC_saver = tf.train.Saver(var_list=model.Dec_params)
    if add_gan == 1:
        DIS_saver = tf.train.Saver(var_list=model.Dis_params)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if FLAGS.enc_restore == True:
        ENC_saver.restore(model.sess, enc_model_dir)
        print("Encoder Model restored in file: %s" % enc_model_dir)
    
    if FLAGS.dec_restore == True:
        DEC_saver.restore(model.sess, dec_model_dir)
        print("Decoder Model restored in file: %s" % dec_model_dir)

    if add_gan == 1:
        if FLAGS.dis_restore == True:
            DIS_saver.restore(model.sess, dis_model_dir)
            print("Discriminator Model restored in file: %s" % dis_model_dir)

    if FLAGS.mode == "train":
        for epoch in range(FLAGS.max_epoch):
            # Shuffle Real Photo every epoch
            img_train_idx     =  shuffle_train_idx(img_train_num)

            for i in (range(int(img_train_iter_num))):

                # Get the batch
                batch_img   = get_batch(img_train, FLAGS.batch_size, img_train_idx , i)
                input_img   = pre_process(batch_img)

                print ("\nEpoch: ", epoch, " / ", FLAGS.max_epoch, " ; Iter: ", i, " / ", int(img_train_iter_num))

                if add_gan == 1:
                    loss_value_enc, loss_value_dec, loss_value_dis, loss_summary = model.update_params(input_img, add_gan)
                    print ("Enc. Loss : ", loss_value_enc, "; Dec. Loss : ", loss_value_dec, "; Dis. Loss : ", loss_value_dis)

                else:
                    loss, loss_summary = model.update_params(input_img, add_gan)
                    print ("Total Loss : %f " % loss)

                model.train_writer.add_summary(loss_summary, i+epoch*img_train_iter_num)
                
                ## Save weight
                if (i+epoch*int(img_train_iter_num))% FLAGS.save_model_freq == FLAGS.save_model_freq-1:
                    if FLAGS.enc_store == True:
                        ENC_save_path = ENC_saver.save(model.sess, model_dir + '/Enc.ckpt')
                        print("Encoder Model saved in file: %s" % ENC_save_path)
                    if FLAGS.dec_store == True:
                        DEC_save_path = DEC_saver.save(model.sess, model_dir + '/Dec.ckpt')
                        print("Decoder Model saved in file: %s" % DEC_save_path)
                    if add_gan == 1:
                        if FLAGS.dis_store == True:
                            DIS_save_path = DIS_saver.save(model.sess, model_dir + '/Dis.ckpt')
                            print("Discriminator Model saved in file: %s" % DIS_save_path)

                if i%FLAGS.save_img_freq==0:
                    print ('\nsaving compact images')
                    #compact_training_img = model.compact_batch_img(batch_img[0:8],batch_img[8:16],batch_img[16:24])

                    imgs_folder = os.path.join(img_dir, 'imgs')
                    if not os.path.exists(imgs_folder):
                        os.makedirs(imgs_folder)
                    
                    imgnum = i/FLAGS.save_img_freq+img_train_iter_num/FLAGS.save_img_freq*epoch
                    #imsave(os.path.join(imgs_folder, 'training_img_iter%d.png') % imgnum, compact_training_img)
                    model.generate_and_save_compact_images(FLAGS.batch_size, input_img, img_dir, imgnum)