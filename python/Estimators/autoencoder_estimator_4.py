# Weishan Version June 20. Dataset should be generated before training and remain unchanged for one user.
import time
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import numpy as np
from Estimators.map_estimator import BemMapEstimator
from Models.net_models import Autoencoder
from joblib import Parallel, delayed
import multiprocessing
import pickle
import pdb
import os
import scipy.io
from copy import deepcopy

# AO:
import gc
import datetime


class AutoEncoderEstimatorFed(BemMapEstimator):
    """
       Arguments:
           n_grid_points_x : number of grid points along x-axis
           n_grid_points_y : number of grid points along y-axis
           add_mask_channel : flag for adding the mask as the second channel at the input of the autoencoder: set to
           False by default.
           transfer_learning : flag for disabling the transfer learning, by default it is set to True.
           learning_rate:  learning rate
       """

    def __init__(self,
                 n_pts_x=32,
                 n_pts_y=32,
                 arch_id='8',
                 c_length=64,
                 n_filters=32,
                 activ_func_name=None,
                 est_separately_accr_freq=False,
                 add_mask_channel=True,
                 use_masks_as_tensor=False,
                 weight_file=None,
                 load_all_weights=None,
                 save_as='weights.h5',
                 n_walkers=2,
                 **kwargs):
        super(AutoEncoderEstimatorFed, self).__init__(**kwargs)
        self.str_name = "Proposed"
        self.n_grid_points_x = n_pts_x
        self.n_grid_points_y = n_pts_y
        self.code_length = c_length
        self.add_mask_channel = add_mask_channel
        self.use_masks_as_tensor = use_masks_as_tensor
        self.n_filters = n_filters
        self.activ_func_name = activ_func_name
        self.est_separately_accr_freq = est_separately_accr_freq
        architecture_name = 'convolutional_autoencoder_%s' % arch_id
        self.n_walkers = n_walkers
        #gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)

        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

        #config = tf.compat.v1.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.9
        #session = tf.compat.v1.Session(config=config)

        # self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():

        # Retain "self.chosen_model" temporarily so we don't have to change too much code
        self.chosen_model = getattr(Autoencoder(height=self.n_grid_points_x, #run the func called 'architecture_name' in obj 'Autoencoder' to build and return DNN  
                                                width=self.n_grid_points_y,
                                                c_len=self.code_length,
                                                add_mask_channel=self.add_mask_channel,
                                                mask_as_tensor=self.use_masks_as_tensor,
                                                bases=self.bases_vals,
                                                n_filters=self.n_filters,
                                                activ_function_name=self.activ_func_name,
                                                est_separately_accr_freq=self.est_separately_accr_freq), architecture_name)() 
        self.chosen_model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(),loss='mse',sample_weight_mode='temporal', weighted_metrics=[]) #resolve a warning
        # self.chosen_model.compile(optimizer=tf.compat.v1.train.GradientDescentOptimizer(),loss='mse',sample_weight_mode='temporal', weighted_metrics=[]) #resolve a warning

            # AO DIDN'T USE: self.chosen_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.0),loss='mse',sample_weight_mode='temporal')
        # self.chosen_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2, momentum=0.9),
        #                           loss= tf.keras.losses.mean_squared_error, #'mse',
        #                           sample_weight_mode='temporal',
        #                           weighted_metrics=[])
        # self.chosen_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9, decay = 1e-5),
        #                                   loss=tf.keras.losses.mean_squared_error,  # 'mse',
        #                                   sample_weight_mode='temporal',
        #                                   weighted_metrics=[])
        #self.chosen_model.compile(optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-6),loss='mse', sample_weight_mode='temporal')
        # AO: Remember, the learning rate isn't actually set here! It's set in autoencoder_experiments


        self.chosen_models = []
        for w_ind in range(self.n_walkers):
            self.chosen_models.append(getattr(Autoencoder(height=self.n_grid_points_x,
                                                width=self.n_grid_points_y,
                                                c_len=self.code_length,
                                                add_mask_channel=self.add_mask_channel,
                                                mask_as_tensor=self.use_masks_as_tensor,
                                                bases=self.bases_vals,
                                                n_filters=self.n_filters,
                                                activ_function_name=self.activ_func_name,
                                                est_separately_accr_freq=self.est_separately_accr_freq), architecture_name)())
            self.chosen_models[w_ind].compile(optimizer=tf.compat.v1.train.AdamOptimizer(), loss='mse', sample_weight_mode='temporal', weighted_metrics=[])
                # AO DIDN'T USE: self.chosen_models[w_ind].compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.0),loss='mse', sample_weight_mode='temporal')
            # self.chosen_models[w_ind].compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2, momentum=0.9),
            #                                   loss= tf.keras.losses.mean_squared_error, #'mse',
            #                                   sample_weight_mode='temporal',
            #                                   weighted_metrics=[])
            # self.chosen_models[w_ind].compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9, decay = 1e-5),
            #                                   loss= tf.keras.losses.mean_squared_error, #'mse',
            #                                   sample_weight_mode='temporal',
            #                                   weighted_metrics=[])
            #self.chosen_models[w_ind].compile(optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-6),loss='mse', sample_weight_mode='temporal')

        # Copy over the weights from self.chosen_model to "initialize" the other models
        for w_ind in range(self.n_walkers):
            for ee in range(len(self.chosen_model.layers)):
                self.chosen_models[w_ind].layers[ee].set_weights(self.chosen_model.layers[ee].get_weights())

        if weight_file:
            self.chosen_model.load_weights(weight_file)

        if weight_file:
            delim = '/'
            sp = weight_file.split(delim)
            nname = sp[-1]

            # Load the federated weights, if desired
            if load_all_weights:
                for w_ind in range(self.n_walkers):
                    if (nname[-3:] == '.h5'):
                        delim = '.'
                        sp = nname.split(delim)
                        fname = sp[0] + '_' + str(w_ind) + delim + sp[1]
                        self.chosen_models[w_ind].load_weights('output/autoencoder_experiments/savedWeights/' + fname)
                    else:
                        print("AO: HANDLE THIS WEIGHT LOADING CASE!!!")


        # self.chosen_model.summary()
        # quit()

        if save_as:
            self.save_as = save_as

    def estimate_map(self, sampled_map, mask, meta_data):
        """

        :param sampled_map:  the sampled map with incomplete entries, 2D array with shape (n_grid_points_x,
                                                                                           n_grid_points_y)
        :type sampled_map: float
        :param mask: is a binary array of the same size as the sampled map:
        :return: the reconstructed map,  2D array with the same shape as the sampled map
        """
        if self.add_mask_channel:
            sampled_map_exp = np.expand_dims(sampled_map, axis=0)
            mask_exp = np.expand_dims(np.expand_dims(mask, axis=0), axis=3)
            meta_exp = np.expand_dims(np.expand_dims(meta_data, axis=0), axis=3)
            if self.use_masks_as_tensor:
                # add the masks as a tensor
                sampled_map_feed = np.concatenate((sampled_map_exp, mask_exp, -meta_exp), axis=3)
            else:
                # combine masks into a single matrix
                sampled_map_feed = np.concatenate((sampled_map_exp, mask_exp - meta_exp), axis=3)
        else:
            sampled_map_feed = np.expand_dims(sampled_map, axis=0)
        reconstructed = self.chosen_model.predict(x=sampled_map_feed)

        return np.reshape(reconstructed[0, :, :], sampled_map.shape)

    def estimate_bem_coefficient_map(self, sampled_map, mask, meta_data):

        # obtain coefficients from the autoencoder estimator
        if self.add_mask_channel:
            sampled_map_exp = np.expand_dims(sampled_map, axis=0)
            mask_exp = np.expand_dims(np.expand_dims(mask, axis=0), axis=3)
            meta_exp = np.expand_dims(np.expand_dims(meta_data, axis=0), axis=3)
            if self.use_masks_as_tensor:
                # add the masks as a tensor
                sampled_map_feed = np.concatenate((sampled_map_exp, mask_exp, -meta_exp), axis=3)
            else:
                # combine masks into a single matrix
                sampled_map_feed = np.concatenate((sampled_map_exp, mask_exp - meta_exp), axis=3)
        else:
            sampled_map_feed = np.expand_dims(sampled_map, axis=0)
        estimated_coeffs = self.get_autoencoder_coefficients_dec(sampled_map_feed)
        return estimated_coeffs[0]

    def train(self,
              generator,
              sampler,
              learning_rate=1e-1,
              n_super_batches=1,
              **kwargs):
        hist = []
        latent_vars = []
        nsb_range = range(n_super_batches)
        for ind_s_batch in range(n_super_batches):
            l_rate = learning_rate / (2 ** ind_s_batch)
            history, latent_vars = self.train_one_batch(generator, sampler, l_rate, **kwargs)
            print("Superbatch " + str(ind_s_batch) + " out of " + str(nsb_range[-1]) + " complete.")
            gc.collect()
            a = 1

        return hist, latent_vars


    def train_fed(self,
              generator,
              sampler,
              learning_rate= None, #5e-8,
              n_super_batches=1,
              n_walkers=2,
              n_epochs=10,
              **kwargs):


        # History, n_walkers x 3 x (n_epochs x n_superbatches)
        hist = np.zeros((self.n_walkers, 3, n_epochs*n_super_batches))

        latent_vars = []
        nsb_range = range(n_super_batches)
        self.datasets = [] #list containing local datasets
        for ind_s_batch in range(n_super_batches):
            #l_rate = learning_rate / (2 ** ind_s_batch)  # AO: This was the original code. Also note this can cause an error if there are too many superbatches (>1-2k)
            l_rate = learning_rate * ((1)**(ind_s_batch/100) ) # AO: Slower learning rate reduction
            print('=====****=====learning rate:', l_rate)

            history, latent_vars = self.train_one_batch(generator, sampler, l_rate, n_epochs=n_epochs, sp_batch_idx=ind_s_batch, **kwargs) # AO: 'hist' was temporarily 'history'
            for ind_w in range(self.n_walkers):
                hist[ind_w,:,ind_s_batch*n_epochs:ind_s_batch*n_epochs+n_epochs] = history[ind_w]

            print("Superbatch " + str(ind_s_batch) + " out of " + str(nsb_range[-1]) + " complete.")
            print("Running garbage collection...")
            gc.collect()

            # Check that the weights of the first layer of the first two models, if available, aren't the same.
            if self.n_walkers > 1:
                arr = self.chosen_models[0].trainable_weights[-2].numpy() == self.chosen_models[1].trainable_weights[-2].numpy()
                tval = arr[0][0][0][0]
                if tval: # If the first weight of the first layer of the first two models match, continue
                    result = np.all(arr == tval)
                    if result: # If result is true, all the values are true
                        print('SECOND-TO-LAST LAYER OF THE FIRST TWO MODELS MATCH; THERE MAY BE AN ERROR!!!')

            print("Federated Averaging models for this superbatch...")
            # For each walker, sum the weights and biases, layer by layer
            for w_ind in range(self.n_walkers):
                if (w_ind == 0):
                    # Take "self.chosen_model" and set weights equal to self.chosen_models[0], layer by layer
                    for ee in range(len(self.chosen_model.layers)):
                        #self.chosen_model.trainable_weights[ee].assign(self.chosen_models[w_ind].trainable_weights[ee].numpy())
                        self.chosen_model.layers[ee].set_weights( deepcopy(self.chosen_models[w_ind].layers[ee].get_weights()) )
                else:
                    # Add the weights from this model to "self.chosen_model", layer by layer
                    for ee in range(len(self.chosen_model.layers)):
                        #self.chosen_model.trainable_weights[ee].assign(self.chosen_model.trainable_weights[ee].numpy() + self.chosen_models[w_ind].trainable_weights[ee].numpy())
                        #self.chosen_model.layers[ee].set_weights(self.chosen_model.layers[ee].get_weights() + self.chosen_models[w_ind].layers[ee].get_weights()) # AO: HOW DOES THIS WORK IF BELOW (DIVIDING THE WEIGHTS BY N WALKERS) DOESN'T WORK?
                        z1 = deepcopy( self.chosen_model.layers[ee].get_weights() )
                        z2 = deepcopy( self.chosen_models[w_ind].layers[ee].get_weights() )
                        if z1:  # If the list isn't empty
                            for ff in range(len(z1)):
                                z1[ff] = (z1[ff] + z2[ff]) # / self.n_walkers
                        self.chosen_model.layers[ee].set_weights(z1)


            # Average the model weights, layer by layer
            for ee in range(len(self.chosen_model.layers)):
                cmod = deepcopy( self.chosen_model.layers[ee].get_weights() )
                if cmod: # If the list isn't empty
                    for ff in range(len(cmod)):
                        cmod[ff] = cmod[ff] / self.n_walkers
                self.chosen_model.layers[ee].set_weights(cmod)



            print("Sending weights to server...")
            for w_ind in range(self.n_walkers):
                for ee in range(len(self.chosen_model.layers)):
                    self.chosen_models[w_ind].layers[ee].set_weights( deepcopy(self.chosen_model.layers[ee].get_weights()) )


        return hist, latent_vars

    def train_one_batch(self,
                        generator,
                        sampler,
                        l_rate,
                        n_maps=128,
                        perc_train=0.9,
                        v_split_frac=1,
                        n_resamples_per_map=1,
                        n_epochs=1,
                        batch_size=64,
                        sp_batch_idx=0,
                        enable_noisy_targets=False,
                        l_fraction_maps_from_each_sampler=None):

        """Operation: each walker train on 1 super_batch for n_epochs epochs, no fedavg inside. each map was originally freshly-generated
        ARGUMENTS:
        'sp_batch_idx': super batch index. if 0 then generate data and save. If positive then use generated data
        `generator`: object descending from class MapGenerator
        `sampler` : object of class Sampler or list of objects of class Sampler.
        `v_split_frac` : if 1, then targets are the entire maps. If tuple or list of length 2, then two splits of the
        sampled maps are used as input and target. The input contains a fraction v_split_frac[0]  of the samples
        whereas the target contains a fraction v_split_frac[1]  of the samples.
        `l_fraction_maps_from_each_sampler` : list of float between 0 and 1 that add up to 1. The n-th entry indicates
        the fraction of maps to be sampled with `sampler[n]`. If set to None, an equal number of maps are
        sampled with each sampler in `sampler`.   [FUTURE, OPTIONAL]
        """

        # This function returns 'history' and 'codes'
        # These should be lists, as opposed to numpy arrays.
            # function to generate training points
        def process_one_map(ind_map, pt_map = None, pm_meta_map = None, pt_ch_power = None, pt_sampled_map=None, pm_mask=None):
            """
            Retuns a list of data points  where one data point is a dictionary
            with keys "x_point", "y_point", "y_mask", and "channel_power_map"

            """
            if pt_map is None: # if not prepared, generate one; if prepraed, just use it
                t_map, m_meta_map, t_ch_power = generator.generate()
                print('+++ ++ ++ ++ check raw map generated:',t_map.max(),t_map.min())
            else:
                t_map = pt_map
                m_meta_map = pm_meta_map
                t_ch_power = pt_ch_power

            if pt_sampled_map is None:
                t_sampled_map, m_mask = sampler.sample_map(t_map, m_meta_map)
            else:
                t_sampled_map = pt_sampled_map
                m_mask = pm_mask

            t_sampled_map, m_mask = sampler.sample_map(t_map, m_meta_map)
            if v_split_frac == 1:

                # m_mask_and_meta = m_mask - m_meta_map
                m_mask_out = 1 - m_meta_map

                # reshaping and adding masks
                t_sampled_map_in, v_map_out, v_mask_out = self.format_preparation(t_sampled_map, m_meta_map, m_mask,
                                                                                  t_map, m_mask_out)

                data_point = {"x_point": t_sampled_map_in,  # Nx X Ny X Nf (+1)
                              "y_point": v_map_out,  # Nx Ny Nf
                              "y_mask": v_mask_out,  # Nx Ny Nf
                              "channel_power_map": t_ch_power}  # Nx X Ny X B
                l_data_points = [data_point]

            elif len(v_split_frac) == 2:
                l_data_points = []
                for ind_resample in range(n_resamples_per_map):
                    # resample the sampled map
                    t_sampled_map_in, m_mask_in, t_sampled_map_out, m_mask_out = sampler.resample_map(
                        t_sampled_map, m_mask, v_split_frac)

                    # reshaping and adding masks
                    t_sampled_map_in, v_map_out, v_mask_out = self.format_preparation(t_sampled_map_in, m_meta_map, m_mask_in,
                                                                                      t_sampled_map_out, m_mask_out)

                    data_point = {"x_point": t_sampled_map_in,  # Nx X Ny X Nf (+1)
                                  "y_point": v_map_out,  # Nx Ny Nf
                                  "y_mask": v_mask_out,  # Nx Ny Nf
                                  "channel_power_map": t_ch_power}  # Nx X Ny X B
                    l_data_points.append(data_point)

            else:
                Exception("invalid value of v_split_frac")
                l_data_points = None
            return l_data_points
        def ll_to_nparray(str_key):
            return np.array(
                [data_point[str_key] for l_data_points in self.datasets[-1] for data_point in l_data_points])

        # Create a numpy array to store the results of the training and validation loss for each epoch, for each walker
        hist = np.zeros((self.n_walkers, 3, n_epochs))
        
        # Pregenerate data for this loop that all walkers will use after they randomly sample
        # commented to make random true map within one superbatch
        # t_map, m_meta_map, t_ch_power = generator.generate()
        t_sampled_map = None
        m_mask = None
        #=== new item, want to make maps random, related to commenting generator.generate()
        t_ch_power=None
        m_meta_map = None
        t_map =None

        if not sp_batch_idx: #if the first superbatch, need to make data. Save train/valid/mask in lists
            start_time = time.time()

            t_x_points_list = []
            t_y_points_list = []
            t_y_masks_list = []
            self.t_channel_pows_list = []
            self.t_x_points_train_list = []
            t_x_points_valid_list = []
            t_y_points_train_list = []
            t_y_points_valid_list = []
            t_y_masks_train_list = []
            t_y_masks_valid_list = []
            t_channel_pows_train_list = []
            train_xy_list = [] #input, label pairs
            val_xy_list = []
            self.trainset_list = [] #dataset obj list for walkers
            self.valset_list = []
            self.internal_eval_list = []


            for w_ind in range(self.n_walkers): #iterate to make data for walkers
                num_cores = int(multiprocessing.cpu_count() / multiprocessing.cpu_count())
                num_cores = int(multiprocessing.cpu_count())
                self.datasets.append( # Generate all data points using parallel processing
                                Parallel(n_jobs=num_cores, backend='threading')(delayed(process_one_map)(ind_map,t_map, m_meta_map, t_ch_power, t_sampled_map, m_mask)
                                                                              for ind_map in range(int(n_maps)))
                                )
                # from 'datasets', generate training/valid/mask datasets and save in lists
                t_x_points_list.append(ll_to_nparray("x_point"))
                print('check data type:', type( t_x_points_list[0] ), type(t_x_points_list), len(t_x_points_list))
                print('=========check training t_x_points_list tensor sizes:', t_x_points_list[0].shape
                      # t_y_points_train_list[0].shape,
                      # t_y_masks_train_list[0].shape
                      )
                # print('Check dataset module attributes:',dir(tf.data.Dataset))
                t_y_points_list.append(ll_to_nparray("y_point"))
                t_y_masks_list.append(ll_to_nparray("y_mask"))
                self.t_channel_pows_list.append(ll_to_nparray("channel_power_map"))

                overall_n_maps = t_x_points_list[-1].shape[0] # this 'overall' corresponds to current walker
                self.n_maps_train = int(perc_train * overall_n_maps)

                self.t_x_points_train_list.append(t_x_points_list[-1][0:self.n_maps_train])
                t_x_points_valid_list.append(t_x_points_list[-1][self.n_maps_train:])

                t_y_points_train_list.append(t_y_points_list[-1][0:self.n_maps_train])
                t_y_points_valid_list.append(t_y_points_list[-1][self.n_maps_train:])

                t_y_masks_train_list.append(t_y_masks_list[-1][0:self.n_maps_train])
                t_y_masks_valid_list.append(t_y_masks_list[-1][self.n_maps_train:])

                t_channel_pows_train_list.append(self.t_channel_pows_list[-1][0:self.n_maps_train])
                if w_ind == 0: # Check data list shape
                    print('=========check training x, y, mask, tensor sizes:', self.t_x_points_train_list[0].shape,
                          t_y_points_train_list[0].shape,
                          t_y_masks_train_list[0].shape
                    )

                # train_xy_list.append([ (t_x_points_train_list[-1][i], t_y_points_train_list[-1][i], t_y_masks_train_list[-1][i])
                #                        for i in range(n_maps_train) ]) # slicing maynot work
                # val_xy_list.append([ (t_x_points_valid_list[-1][i], t_y_points_valid_list[-1][i], t_y_masks_valid_list[-1][i])
                #                        for i in range(overall_n_maps-n_maps_train) ])
                # list of x/y/mask-tuple version dataset (seems wrong)
                # trainset_list.append( tf.data.Dataset.from_tensor_slices( train_xy_list[-1] ) )  #from_list
                # valset_list.append( tf.data.Dataset.from_tensor_slices( val_xy_list[-1] ) ) #from_list
                # tuple of x/y/mask-list version dataset
                self.trainset_list.append( tf.data.Dataset.from_tensor_slices(
                                                    (self.t_x_points_train_list[-1], t_y_points_train_list[-1],
                                                    t_y_masks_train_list[-1])
                                        ).batch(batch_size).shuffle(buffer_size = self.n_maps_train).cache().prefetch(tf.data.AUTOTUNE)
                                           ) #from_list
                self.valset_list.append( tf.data.Dataset.from_tensor_slices(
                                                    (t_x_points_valid_list[-1], t_y_points_valid_list[-1],
                                                    t_y_masks_valid_list[-1])
                                        ).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
                                         )#from_list
                # if w_ind==0: #Check dataset size
                #     tf.data.experimental.save(
                #         self.trainset_list[0], 'datasize', compression=None, shard_func=None, checkpoint_args=None
                #     )

                # k=0 #check dataset output value shape:
                # for element in self.trainset_list[-1]:
                #     k = k+1
                #     if k<3:
                #         print('============element shapes',element[0].shape,element[1].shape, element[2].shape )

                self.internal_eval_list.append( InternalEvaluation(
                    validation_data=(self.t_x_points_train_list[-1], t_y_points_train_list[-1], t_y_masks_train_list[-1])) )

            end_time = time.time()
            elapsed_time = end_time - start_time
            print('The elapsed time for generating and sampling training and validation maps using multiple processors '
                  'with %d cores is' % num_cores, time.strftime("%H hours %M min %S sec", time.gmtime(elapsed_time))) 
# =================updated training part to work with lists of train/valid/mask
        nw_range = range(self.n_walkers)
        for w_ind in range(self.n_walkers):
            print("Training walker " + str(w_ind) + " out of " + str(nw_range[-1]) + " now.")
            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            self.chosen_models[w_ind].optimizer.learning_rate= l_rate
            #self.chosen_models[w_ind].optimizer.optimizer._lr = 0.01
            # seems to cost some time in the following step
            # train_history = self.chosen_models[w_ind].fit(x=self.trainset_list[w_ind],
            #                                       batch_size=None,
            #                                       epochs=sp_batch_idx+1, #
            #                                       initial_epoch = sp_batch_idx, # not exist in original version
            #                                       validation_data = self.valset_list[w_ind],
            #                                       callbacks=[self.internal_eval_list[w_ind], tensorboard_callback],
            #                                       verbose=2)
            train_history = self.chosen_models[w_ind].fit(x=self.trainset_list[w_ind],
                                                  # y=t_y_points_train,
                                                  batch_size=batch_size, #,
                                                  # sample_weight=t_y_masks_train,
                                                  # epochs=sp_batch_idx+1, #
                                                  epochs=n_epochs, #original version
                                                  # initial_epoch = sp_batch_idx, # not exist in original version
                                                  # validation_data=(
                                                  #     t_x_points_valid, t_y_points_valid, t_y_masks_valid),
                                                  validation_data = self.valset_list[w_ind],
                                                  callbacks=[self.internal_eval_list[w_ind], tensorboard_callback],
                                                  verbose=1)
            # train_history = self.chosen_models[w_ind].fit(x=self.trainset_list[w_ind],
            #                                               # y=t_y_points_train,
            #                                               batch_size=batch_size,  # ,
            #                                               # sample_weight=t_y_masks_train,
            #                                               # epochs=sp_batch_idx+1, #
            #                                               epochs=n_epochs,  # original version
            #                                               # initial_epoch = sp_batch_idx, # not exist in original version
            #                                               # validation_data=(
            #                                               #     t_x_points_valid, t_y_points_valid, t_y_masks_valid),
            #                                               validation_data=self.valset_list[w_ind],
            #                                               callbacks=[self.internal_eval_list[w_ind],
            #                                                          tensorboard_callback],
            #                                               verbose=1)

            # print('================History keys',train_history.history.keys())
            history = np.array([np.array(self.internal_eval_list[w_ind].train_losses),
                                np.array(train_history.history['val_loss']),
                                np.array(train_history.history['loss'])])

            # Save the federated weights
            if(self.save_as[-3:] == '.h5'):
                delim = '.'
                sp = self.save_as.split(delim)
                fname = sp[0] + '_' + str(w_ind) + delim + sp[1]
                self.chosen_models[w_ind].save_weights('output/autoencoder_experiments/savedWeights/' + fname)
            else:
                self.chosen_models[w_ind].save_weights('output/autoencoder_experiments/savedWeights/' + self.save_as)

            # Save the weights of the federated model # is it needed for every local walker?
            self.chosen_model.save_weights('output/autoencoder_experiments/savedWeights/' + self.save_as)

            # AO: Below doesn't seem too relevant
            # obtain the codes for some training maps
            trained_encoder = self.chosen_models[w_ind].get_layer('encoder')
            # trained_encoder.summary()
            # trained_decoder.summary()
            out_layer_ind = len(trained_encoder.layers) - 1
            num_codes_to_show = min(2 * self.n_maps_train, int(3e3))
            codes = get_layer_activations(trained_encoder, self.t_x_points_train_list[w_ind][0: num_codes_to_show], out_layer_ind)

            # save some l_training_coefficients for later checking
            n_saved_maps = 500
            with open('output/autoencoder_experiments/savedResults/True_and_Est_training_bcoeffs.pickle', 'wb') as f_bcoeff:
                pickle.dump([self.t_channel_pows_list[w_ind][0:n_saved_maps], self.t_x_points_train_list[w_ind][0:n_saved_maps]], f_bcoeff)

            hist[w_ind] = history

#==================Old version to be removed==========================
        # nw_range = range(self.n_walkers)
        # for w_ind in range(self.n_walkers):
        #
        #     print("Training walker " + str(w_ind) + " out of " + str(nw_range[-1]) + " now.")
        #
        #     # Get arrays from l_l_data_points
        #
        #     t_x_points = ll_to_nparray("x_point")
        #     t_y_points = ll_to_nparray("y_point")
        #     t_y_masks = ll_to_nparray("y_mask")
        #     t_channel_pows = ll_to_nparray("channel_power_map")
        #
        #     # Training/validation split
        #     overall_n_maps = t_x_points.shape[0]
        #     n_maps_train = int(perc_train * overall_n_maps)
        #
        #     # x_points
        #     t_x_points_train = t_x_points[0:n_maps_train]
        #     t_x_points_valid = t_x_points[n_maps_train:]
        #
        #     # y_points
        #     t_y_points_train = t_y_points[0:n_maps_train]
        #     t_y_points_valid = t_y_points[n_maps_train:]
        #
        #     # y_masks
        #     t_y_masks_train = t_y_masks[0:n_maps_train]
        #     t_y_masks_valid = t_y_masks[n_maps_train:]
        #
        #     # training channel powers
        #     t_channel_pows_train = t_channel_pows[0:n_maps_train]
        #
        #     # Training loss computation using callback
        #     internal_eval = InternalEvaluation(
        #         validation_data=(t_x_points_train, t_y_points_train, t_y_masks_train))
        #
        #     # Tensorflow callback
        #     log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #     tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        #
        #     # Fit
        #     # self.chosen_models[w_ind].optimizer.optimizer._lr = l_rate
        #     self.chosen_models[w_ind].optimizer.learning_rate= l_rate
        #     #self.chosen_models[w_ind].optimizer.optimizer._lr = 0.01
        #     # seems to cost some time in the following step
        #     train_history = self.chosen_models[w_ind].fit(x=t_x_points_train,
        #                                           y=t_y_points_train,
        #                                           batch_size=batch_size,
        #                                           sample_weight=t_y_masks_train,
        #                                           epochs=sp_batch_idx+1, #
        #                                           #epochs=n_epochs, #original version
        #                                           initial_epoch = sp_batch_idx, # not exist in original version
        #                                           validation_data=(
        #                                               t_x_points_valid, t_y_points_valid, t_y_masks_valid),
        #                                           callbacks=[internal_eval, tensorboard_callback],
        #                                           verbose=2)
        #     # print('================History keys',train_history.history.keys())
        #     history = np.array([np.array(internal_eval.train_losses),
        #                         np.array(train_history.history['val_loss']),
        #                         np.array(train_history.history['loss'])])
        #
        #     # Save the federated weights
        #     if(self.save_as[-3:] == '.h5'):
        #         delim = '.'
        #         sp = self.save_as.split(delim)
        #         fname = sp[0] + '_' + str(w_ind) + delim + sp[1]
        #         self.chosen_models[w_ind].save_weights('output/autoencoder_experiments/savedWeights/' + fname)
        #     else:
        #         self.chosen_models[w_ind].save_weights('output/autoencoder_experiments/savedWeights/' + self.save_as)
        #
        #     # Save the weights of the federated model # is it needed for every local walker?
        #     self.chosen_model.save_weights('output/autoencoder_experiments/savedWeights/' + self.save_as)
        #
        #     # AO: Below doesn't seem too relevant
        #     # obtain the codes for some training maps
        #     trained_encoder = self.chosen_models[w_ind].get_layer('encoder')
        #     # trained_encoder.summary()
        #     # trained_decoder.summary()
        #     out_layer_ind = len(trained_encoder.layers) - 1
        #     num_codes_to_show = min(2 * n_maps_train, int(3e3))
        #     codes = get_layer_activations(trained_encoder, t_x_points_train[0: num_codes_to_show], out_layer_ind)
        #
        #     # save some l_training_coefficients for later checking
        #     n_saved_maps = 500
        #     with open('output/autoencoder_experiments/savedResults/True_and_Est_training_bcoeffs.pickle', 'wb') as f_bcoeff:
        #         pickle.dump([t_channel_pows_train[0:n_saved_maps], t_x_points_train[0:n_saved_maps]], f_bcoeff)
        #
        #     hist[w_ind] = history

        # This is the end of "for w_ind in range(self.n_walkers):"
        return hist, codes

    def format_preparation(self,
                           t_input,
                           meta_data,
                           m_mask_in,
                           t_output,
                           m_mask_out,
                           ):
        """
        Returns:
        `t_input_proc`: Nx x Ny x Nf (+1) tensor
        `v_output`: Nx Ny Nf vector (vectorized Nx x Ny x Nf tensor).
        `v_weight_out`: vector of the same dimension as `v_output`.

        """

        if self.add_mask_channel:
            m_mask_exp = np.expand_dims(m_mask_in, axis=2)
            m_meta_exp = np.expand_dims(meta_data, axis=2)
            if self.use_masks_as_tensor:
                # add the masks as a tensor
                t_input_proc = np.concatenate((t_input, m_mask_exp, - m_meta_exp), axis=2)
            else:
                # combine masks into a single matrix
                t_input_proc = np.concatenate((t_input, m_mask_exp - m_meta_exp), axis=2)
        else:
            t_input_proc = t_input

        v_output = np.expand_dims(np.ndarray.flatten(t_output), axis=1)
        t_mask_out = np.repeat(m_mask_out[:, :, np.newaxis], t_output.shape[2], axis=2)
        v_weight_out = np.ndarray.flatten(t_mask_out)

        return t_input_proc, v_output, v_weight_out

    def get_autoencoder_coefficients_dec(self,  t_input):
        encoder = self.chosen_model.get_layer('encoder')
        decoder = self.chosen_model.get_layer('decoder')
        enc_out_layer_ind = len(encoder.layers) - 1
        ind_layer_dec = len(decoder.layers) - 9  # subtract 9 layers to reach the layer giving the coefficients
        codes = get_layer_activations(encoder, t_input, enc_out_layer_ind)
        coeff = get_layer_activations(decoder, codes, ind_layer_dec)
        return coeff  # [:, :, :, 0:(self.bases_vals.shape[0] - 1)] -1 to remove the noise base if it is included




class InternalEvaluation(Callback):
    def __init__(self, validation_data=()):
        super(Callback, self).__init__()
        self.X_data, self.Y_data, self.weights = validation_data

    def on_train_begin(self, logs={}):
        self.train_losses = []

    def on_epoch_end(self, epoch, logs={}):
        train_loss = self.model.evaluate(x=self.X_data, y=self.Y_data, sample_weight=self.weights, batch_size=64,
                                         verbose=1)
        self.train_losses.append(train_loss)
        print("Internal evaluation - epoch: {:d} - loss: {:.6f}".format(epoch, train_loss))


        result = np.all(self.X_data == np.nan)
        if result:
            print('X-data has NaN value!!!')

        result = np.all(self.Y_data == np.nan)
        if result:
            print('Y-data has NaN value!!!')

        result = np.all(self.weights == np.nan)
        if result:
            print('Weights has NaN value!!!')




        return self.train_losses


def get_layer_activations(network, m_input, ind_layer):
    f_get_layer_activations = K.function([network.layers[0].input],
                                         [network.layers[ind_layer].output])
    layer_activations = f_get_layer_activations(m_input)[0]
    return layer_activations



