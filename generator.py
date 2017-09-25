import os
from scipy.misc import imsave
import numpy as np

class Generator(object):

    def update_params(self, input_tensor):
        '''Update parameters of the network

        Args:
            input_tensor: a batch of flattened images

        Returns:
            Current loss value
        '''
        raise NotImplementedError()

    def generate_and_save_images(self, num_samples, directory):
        '''Generates the images using the model and saves them in the directory

        Args:
            num_samples: number of samples to generate
            directory: a directory to save the images
        '''
        imgs = self.sess.run(self.sampled_tensor)
        imgs = np.multiply(imgs, 128.)
        imgs = np.add(imgs, 128.)
        #print (imgs.shape)
        for k in range(imgs.shape[0]):
            imgs_folder = os.path.join(directory, 'imgs')
            if not os.path.exists(imgs_folder):
                os.makedirs(imgs_folder)

            imsave(os.path.join(imgs_folder, '%d.png') % k,
                   imgs[k].reshape(64, 64, 3))


    def generate_and_save_compact_images(self, numsamples, input_imgs, directory, imgnum):
        sampled_imgs       = self.sess.run(self.sampled_tensor)
        sampled_imgs       = np.add(np.multiply(sampled_imgs, 128.), 128.)
        sampled_imgs.reshape(numsamples, 64, 64, 3)
        decoded_input_imgs = self.sess.run(self.DEC_of_ENC, {self.input_tensor: input_imgs})
        decoded_input_imgs = np.add(np.multiply(decoded_input_imgs, 128.), 128.)
        decoded_input_imgs.reshape(numsamples, 64, 64, 3)
        input_imgs         = np.add(np.multiply(input_imgs, 128.), 128.)

        compact_img        = self.compact_batch_img(sampled_imgs[0:8],input_imgs[0:8],decoded_input_imgs[0:8])

        imgs_folder = os.path.join(directory, 'imgs')
        if not os.path.exists(imgs_folder):
            os.makedirs(imgs_folder)
        
        imsave(os.path.join(imgs_folder, 'generate_img_iter%d.png') % imgnum, compact_img)

    def save_visual_attri_compact_images(self, numsamples, input_imgs, attri_num, attri_vector, directory, imgnum):
        
        decoded_input_imgs = self.sess.run(self.DEC_of_ENC, {self.input_tensor: input_imgs})
        decoded_input_imgs = np.add(np.multiply(decoded_input_imgs, 128.), 128.)
        decoded_input_imgs.reshape(numsamples, 64, 64, 3)
        decoded_row        = self.compact_to_row(decoded_input_imgs)

        input_imgs         = np.add(np.multiply(input_imgs, 128.), 128.)
        input_row          = self.compact_to_row(input_imgs)

        compact_img        = np.concatenate( (input_row, decoded_row),axis=0)

        for i in range(attri_num):
            with_attri_imgs    = self.sess.run(self.with_attri_tensor, {self.input_tensor: input_imgs, self.visual_attri: attri_vector[:,i]})
            with_attri_imgs    = np.add(np.multiply(with_attri_imgs, 128.), 128.)
            with_attri_imgs.reshape(numsamples, 64, 64, 3)
            with_attri_row     = self.compact_to_row(with_attri_imgs)
            compact_img        = np.concatenate( (compact_img, with_attri_row),axis=0)

        imgs_folder        = os.path.join(directory, 'imgs')
        if not os.path.exists(imgs_folder):
            os.makedirs(imgs_folder)
        
        imsave(os.path.join(imgs_folder, 'with_attri_img_iter%d.png') % imgnum, compact_img)

    def compact_to_row(self, input_npary):
        tmp = input_npary
        row1 = np.concatenate( ( tmp[0,:,:,:], tmp[1,:,:,:], tmp[2,:,:,:], tmp[3,:,:,:], tmp[4,:,:,:], tmp[5,:,:,:], tmp[6,:,:,:], tmp[7,:,:,:]),axis=1)
        return row1

    def compact_batch_img(self,input_npary,input_npary2,input_npary3):
        #input1,2,3 : [-0.5,0.5] rgb or gray
        '''
        if input_npary.shape[3]==1:
            tmp = np.concatenate( (input_npary, input_npary, input_npary), axis=3)
        else:
            tmp = input_npary
        tmp  = np.multiply(tmp, 128.)         
        tmp  = np.add(tmp, 128.)

        if input_npary2.shape[3]==1:
            tmp2 = np.concatenate( (input_npary2, input_npary2, input_npary2), axis=3)
        else:
            tmp2 = input_npary2
        tmp2  = np.multiply(tmp2, 128.)         
        tmp2  = np.add(tmp2, 128.)

        if input_npary3.shape[3]==1:
            tmp3 = np.concatenate( (input_npary3, input_npary3, input_npary3), axis=3)
        else:
            tmp3 = input_npary3
        tmp3  = np.multiply(tmp3, 128.)         
        tmp3  = np.add(tmp3, 128.)
        '''
        tmp = input_npary
        tmp2 = input_npary2
        tmp3 = input_npary3
        row1 = np.concatenate( ( tmp[0,:,:,:], tmp[1,:,:,:], tmp[2,:,:,:], tmp[3,:,:,:], tmp[4,:,:,:], tmp[5,:,:,:], tmp[6,:,:,:], tmp[7,:,:,:]),axis=1)
        row2 = np.concatenate( (tmp2[0,:,:,:],tmp2[1,:,:,:],tmp2[2,:,:,:],tmp2[3,:,:,:],tmp2[4,:,:,:],tmp2[5,:,:,:],tmp2[6,:,:,:],tmp2[7,:,:,:]),axis=1)
        row3 = np.concatenate( (tmp3[0,:,:,:],tmp3[1,:,:,:],tmp3[2,:,:,:],tmp3[3,:,:,:],tmp3[4,:,:,:],tmp3[5,:,:,:],tmp3[6,:,:,:],tmp3[7,:,:,:]),axis=1)

        compact_img = np.concatenate( (row1,row2,row3),axis=0)
        return compact_img