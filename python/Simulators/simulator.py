import numpy as np
from numpy import linalg as npla
from joblib import Parallel, delayed
import multiprocessing
import random
import time


class Simulator:
    """
       Arguments:
              n_runs: the number of monte carlo runs

    """

    def __init__(self,
                 n_runs=1,
                 use_parallel_proc=False):

        self.n_runs = n_runs
        self.use_parallel_proc = use_parallel_proc

    def simulate(self,
                 generator,
                 sampler,
                 estimator,
                 ):
        """
        :param generator:  generator for generating maps used for training
        :type generator:   class
        :param sampler:   sampler for sampling the generated maps
        :type sampler:    class
        :param estimator:  estimator that reconstructs the sampled map
        :type estimator: 
        :return:  the estimation error
        :type estimator:
        :return:real_map  consider_shadowing
        :rtype: float
        """
        def process_one_run(ind_run): #t_map:48x48x1, m_meta_map: 48x48, m_meta_map_all_freqs:48x48x1  t_sampled_map_allfreq: 48x48x1?
            t_map, m_meta_map, _ = generator.generate()
            m_meta_map_all_freqs = np.repeat(m_meta_map[:, :, np.newaxis], t_map.shape[2], axis=2)
            t_sampled_map_allfreq, mask = sampler.sample_map(t_map, m_meta_map)
            # start_time = time.time()
            t_estimated_map = estimator.estimate_map(t_sampled_map_allfreq, mask, m_meta_map)
            # end_time = time.time()
            # elapsed_time = end_time - start_time
            # print('The run-time of the estimator XX  is %.5f' % elapsed_time),
            v_meta = m_meta_map_all_freqs.flatten()
            v_map = t_map.flatten()
            v_est_map = t_estimated_map.flatten()
            sq_err_one_runs = (npla.norm((1 - v_meta) * (v_map - v_est_map))) ** 2 / len(
                np.where(v_meta == 0)[0]) #use [0] cuz the output is a tuple
            return sq_err_one_runs

        if self.use_parallel_proc:
            num_cores = int(multiprocessing.cpu_count() / 2)
            sq_err_all_runs = Parallel(n_jobs=num_cores)(delayed(process_one_run)(i)
                                                         for i in range(self.n_runs))
            sq_err_all_runs_arr = np.array(sq_err_all_runs)
        else:
            sq_err_all_runs_arr = np.zeros((1, self.n_runs))
            for ind_run in range(self.n_runs):
                sq_err_all_runs_arr[0, ind_run] = process_one_run(ind_run)
        rmse = np.sqrt(np.mean(sq_err_all_runs_arr))
        return rmse

class SimulatorNew:
    """
        Try to parallel, failed
        Try to improve efficiency through batch processing (feed batch to DNN but calculate error in loop)
        Some old code/functions remained
       Arguments:
              n_runs: the number of monte carlo runs

    """

    def __init__(self,
                 n_runs=1,
                 use_parallel_proc=False,
                 generator=None,
                 sampler=None,
                 sampling_factors=None,
                 btsz = 512): #batchsize

        self.n_runs = n_runs
        self.use_parallel_proc = use_parallel_proc
        self.btsz = btsz
        self.generator = generator
        self.sampler = sampler
        # sampler.v_sampling_factor =
        self.sampling_factors = sampling_factors #list
        # ==== generate dataset for simulator with sampling_factor and n_runs each sampling rate ====
        self.data_dic = {a: None for a in sampling_factors } # prepare testing data of diff sampling rate, one big ndarray for sampling_rate, n_runsx48x48x3

        # 1. Loop to generate n_runs true_maps and save in a list. 2. Loop to sample len(sampling_factor) testsets from true_map
        # self.True_maps is a list containing n_runs tuples "( t_map 48x48x1, m_meta_map 48x48, _ )"
        num_cores = int(multiprocessing.cpu_count())
        self.True_maps = Parallel(n_jobs=num_cores, backend='threading')(delayed(self.generator.generate)() for _ in range(n_runs) )
        # mshape = self.True_maps[0][0].shape #48x48x1
        shap2 = self.True_maps[0][0].shape[2] #shap2=1, (48,48,1)
        # self.True_t_map = [ np.expand_dims(TrueMap[0].flatten(), axis=0) for TrueMap in self.True_maps] #flattened true maps in a list without mask/meta_map
        # self.TrueMapArray = np.concatenate( #label(true map) for calculate RMSE, n_runsx2304
        #     [np.expand_dims(TrueMap[0].flatten(), axis=0) for TrueMap in self.True_maps] #flattened true maps in a list without mask/meta_map
        #     ,axis=0)
        self.TrueMapArray = np.array( #label(true map) for calculate RMSE, n_runs x 2304 x 1
            [np.expand_dims(TrueMap[0].flatten(), axis=1) for TrueMap in self.True_maps]) #flattened true maps in a list without mask/meta_map
            # ,axis=0)
        print('True maps prepared')

        # Save the flattened version of meta_map, the error in buildings is ignored.
        metamaplist = [np.expand_dims(np.repeat(MetaMap[1][:, :, np.newaxis], shap2, axis=2).flatten(), axis=1) for
                         MetaMap in self.True_maps]# list of (2304,1)
        self.MetaMapArray = np.array(metamaplist) # (n_runs, 2304, 1)
        print('City maps prepared')

        # m_meta_map_all_freqs = np.repeat(m_meta_map[:, :, np.newaxis], t_map.shape[2], axis=2)
        # v_meta = m_meta_map_all_freqs.flatten()

        for a in sampling_factors: # handle the input data for each sampling factor
            self.sampler.v_sampling_factor = a
            # t_map, m_meta_map, _ = generator.generate()
            sampled_a = Parallel(n_jobs=num_cores, backend='threading')(delayed(self.sampler.sample_map)(self.True_maps[nn][0], self.True_maps[nn][1]) for nn in range(int(n_runs)) )
            # parallel sampler got list of tuples "(t_sampled_map_allfreq 48x48x1, mask 48x48)"
            self.data_dic[a] = np.empty((self.n_runs, 48,48,2 ), dtype=self.True_maps[0][0].dtype)

            for i, smlpd in enumerate(sampled_a): # save inputs of one sampling factor in an ndarray
                # self.data_dic[a][i] = np.concatenate(
                #     (
                #         np.expand_dims(smlpd[0],axis=0),
                #         np.expand_dims(np.expand_dims(smlpd[1],axis=0), axis=3), #sampled mask
                #         np.expand_dims(np.expand_dims(-1*self.True_maps[i][1],axis=0), axis=3), # meta map
                #         )
                #         , axis=3
                # )

                self.data_dic[a][i] = np.concatenate(
                    (
                        np.expand_dims(smlpd[0],axis=0),
                        np.expand_dims(np.expand_dims(smlpd[1],axis=0), axis=3) - #sampled mask
                        np.expand_dims(np.expand_dims(-1*self.True_maps[i][1],axis=0), axis=3), # meta map
                        )
                        , axis=3
                )
            print( a, 'sampling factor data prepared.')

            # self.estimator.chosen_model.predict(x=self.data_dic[ ], batch_size=1024)
            # stack in to tensor
            # t_sampled_map_allfreq, mask = sampler.sample_map(t_map, m_meta_map)
        # for mapn in range(len(True_maps)): #sample map according to
        # l_l_data_points = Parallel(n_jobs=num_cores, backend='threading')(delayed(self.process_one_map)(ind_map=None) for ind_map in range(int(n_runs)))

    # def process_one_run(self, ind_run, generator, sampler, estimator):
    #     t_map, m_meta_map, _ = generator.generate()
    #     m_meta_map_all_freqs = np.repeat(m_meta_map[:, :, np.newaxis], t_map.shape[2], axis=2)
    #     t_sampled_map_allfreq, mask = sampler.sample_map(t_map, m_meta_map)
    #     # start_time = time.time()
    #     t_estimated_map = estimator.estimate_map(t_sampled_map_allfreq, mask, m_meta_map)
    #     # end_time = time.time()
    #     # elapsed_time = end_time - start_time
    #     # print('The run-time of the estimator XX  is %.5f' % elapsed_time),
    #     v_meta = m_meta_map_all_freqs.flatten()
    #     v_map = t_map.flatten()
    #     v_est_map = t_estimated_map.flatten()
    #     sq_err_one_runs = (npla.norm((1 - v_meta) * (v_map - v_est_map))) ** 2 / len(
    #         np.where(v_meta == 0)[0])
    #     return sq_err_one_runs
    # def simulate_old(self,
    #              generator,
    #              sampler,
    #              estimator,
    #              ):
    #     """ Old version with bad parallel processing and slow serial processing
    #     :param generator:  generator for generating maps used for training
    #     :type generator:   class
    #     :param sampler:   sampler for sampling the generated maps
    #     :type sampler:    class
    #     :param estimator:  estimator that reconstructs the sampled map
    #     :type estimator:
    #     :return:  the estimation error
    #     :type estimator:
    #     :return:real_map  consider_shadowing
    #     :rtype: float
    #     """
    #     if self.use_parallel_proc:
    #         num_cores = int(multiprocessing.cpu_count() / 2)
    #         sq_err_all_runs = Parallel(n_jobs=num_cores)(delayed(self.process_one_run)(i,generator, sampler, estimator)
    #                                                      for i in range(self.n_runs))
    #         sq_err_all_runs_arr = np.array(sq_err_all_runs)
    #     else:
    #         sq_err_all_runs_arr = np.zeros((1, self.n_runs))
    #         for ind_run in range(self.n_runs):
    #             sq_err_all_runs_arr[0, ind_run] = self.process_one_run(ind_run, generator, sampler, estimator)
    #     rmse = np.sqrt(np.mean(sq_err_all_runs_arr))
    #     return rmse

    def simulate(self,
                 # generator,
                 # sampler,
                 sampl_rate,
                 estimator,
                 ):
        """
        New version to simulate in batch, centralized case
        :param generator:  generator for generating maps used for training
        :type generator:   class
        :param sampler:   sampler for sampling the generated maps
        :type sampler:    class
        :param estimator:  estimator that reconstructs the sampled map
        :type estimator:
        :return:  the estimation error
        :type estimator:
        :return:real_map  consider_shadowing
        :rtype: float
        """
        sq_err_all_runs_arr = np.zeros((1, self.n_runs)) #initialize mse list
        #===== use DNN to process inputs in batches, save outputs =====
        #===== Calculate error (serial? together? in batch? TF parallel?)
        out = estimator.chosen_model.predict(x=self.data_dic[sampl_rate], batch_size=self.btsz)

        print(sampl_rate, 'sampling factor maps reconstructed!')
        rmse=(npla.norm((1-self.MetaMapArray)* (out - self.TrueMapArray)))**2 / np.sum(1-self.MetaMapArray)
            # sq_err_one_runs = (npla.norm((1 - v_meta) * (v_map - v_est_map))) ** 2 / len(
            #     np.where(v_meta == 0)[0])  # use [0] cuz the output is a tuple

            # self.MetaMapArray
        # rmse = np.sqrt(np.mean(sq_err_all_runs_arr))
        print(sampl_rate, 'sampling factor tested!')
        return rmse

class SimulatorNewFL:
    """
        Try to improve efficiency through batch processing (feed batch to DNN but calculate error in loop)
        Some old code/functions remained
       Arguments:
              n_runs: the number of monte carlo runs
    """

    def __init__(self,
                 n_runs=1,
                 use_parallel_proc=False,
                 generator=None,
                 sampler=None,
                 sampling_factors=None,
                 btsz = 512): #batchsize

        self.n_runs = n_runs
        self.use_parallel_proc = use_parallel_proc
        self.btsz = btsz
        self.generator = generator
        self.sampler = sampler
        self.sampling_factors = sampling_factors #list
        # ==== generate dataset for simulator with sampling_factor and n_runs each sampling rate ====
        self.data_dic = {a: None for a in sampling_factors } # prepare testing data of diff sampling rate, one big ndarray for sampling_rate, n_runsx48x48x3

        # 1. Loop to generate n_runs true_maps and save in a list. 2. Loop to sample len(sampling_factor) testsets from true_map
        # self.True_maps is a list containing n_runs tuples "( t_map 48x48x1, m_meta_map 48x48, __ )"
        num_cores = int(multiprocessing.cpu_count())
        self.True_maps = Parallel(n_jobs=num_cores, backend='threading')(delayed(self.generator.generate)() for _ in range(n_runs) )
        shap2 = self.True_maps[0][0].shape[2] #shap2=1, (48,48,1)
        print('True maps generated')
        # self.True_t_map = [ np.expand_dims(TrueMap[0].flatten(), axis=0) for TrueMap in self.True_maps] #flattened true maps in a list without mask/meta_map
        # self.TrueMapArray = np.concatenate( #label(true map) for calculate RMSE, n_runsx2304
        #     [np.expand_dims(TrueMap[0].flatten(), axis=0) for TrueMap in self.True_maps] #flattened true maps in a list without mask/meta_map
        #     ,axis=0)
        self.TrueMapArray = np.array( #label(true map) for calculate RMSE, n_runs x 2304 x 1
            [np.expand_dims(TrueMap[0].flatten(), axis=1) for TrueMap in self.True_maps]) #flattened true maps in a list without mask/meta_map
            # ,axis=0)
        print('True maps flattened')

        # Save the flattened version of meta_map, the error in buildings is ignored.
        metamaplist = [np.expand_dims(np.repeat(true_map[1][:, :, np.newaxis], shap2, axis=2).flatten(), axis=1) for
                         true_map in self.True_maps]# list of (2304,1)
        self.MetaMapArray = np.array(metamaplist) # (n_runs, 2304, 1)
        print('Weights for loss calculation prepared')

        # m_meta_map_all_freqs = np.repeat(m_meta_map[:, :, np.newaxis], t_map.shape[2], axis=2)
        # v_meta = m_meta_map_all_freqs.flatten()

        for a in sampling_factors: # handle the input data for each sampling factor
            self.sampler.v_sampling_factor = a
            # parallel sampler got list of tuples "(t_sampled_map_allfreq 48x48x1, mask 48x48)"
            sampled_a = Parallel(n_jobs=num_cores, backend='threading')(delayed(self.sampler.sample_map)(self.True_maps[nn][0], self.True_maps[nn][1]) for nn in range(int(n_runs)) )

            self.data_dic[a] = np.empty((self.n_runs, 48,48,2 ), dtype=self.True_maps[0][0].dtype)

            for i, smlpd in enumerate(sampled_a): # save inputs of one sampling factor in an ndarray
                # self.data_dic[a][i] = np.concatenate( # This version is for separated meta_map and sensor masks.
                #     (
                #         np.expand_dims(smlpd[0],axis=0),
                #         np.expand_dims(np.expand_dims(smlpd[1],axis=0), axis=3), #sampled mask
                #         np.expand_dims(np.expand_dims(-1*self.True_maps[i][1],axis=0), axis=3), # meta map
                #         )
                #         , axis=3
                # )
                self.data_dic[a][i] = np.concatenate( # sensor_mask - meta_map, sharing a channel
                    (
                        np.expand_dims(smlpd[0],axis=0),
                        np.expand_dims(np.expand_dims(smlpd[1],axis=0), axis=3) - #sampled mask
                        np.expand_dims(np.expand_dims(-1*self.True_maps[i][1],axis=0), axis=3), # meta map
                        )
                        , axis=3
                )
            print( a, 'sampling factor data prepared.')

            # self.estimator.chosen_model.predict(x=self.data_dic[ ], batch_size=1024)
            # stack in to tensor
            # t_sampled_map_allfreq, mask = sampler.sample_map(t_map, m_meta_map)
        # for mapn in range(len(True_maps)): #sample map according to
        # l_l_data_points = Parallel(n_jobs=num_cores, backend='threading')(delayed(self.process_one_map)(ind_map=None) for ind_map in range(int(n_runs)))

    def simulate(self,
                 # generator,
                 # sampler,
                 sampl_rate,
                 estimator,
                 ):
        """
        New version to simulate in batch, FL case
        :param generator:  generator for generating maps used for training
        :type generator:   class
        :param sampler:   sampler for sampling the generated maps
        :type sampler:    class
        :param estimator:  estimator that reconstructs the sampled map
        :type estimator:
        :return:  the estimation error
        :type estimator:
        :return:real_map  consider_shadowing
        :rtype: float
        """
        #===== use DNN to process inputs in batches, save outputs =====
        # out = estimator.chosen_model.predict(x=self.data_dic[sampl_rate], batch_size=self.btsz)
        # For single user, zero-pad and cut the self.data_dic[sampl_rate]
        ovp = estimator.overlap
        pad_width = ((0,0),(ovp, ovp), (ovp, ovp), (0, 0))
        data_dict_pd = np.pad(self.data_dic[sampl_rate], pad_width=pad_width, mode='constant', constant_values=0)
        # Feed in local user and get vector output (btsz, 400,1)
        CellWid = estimator.n_grid_points_x  # cell width, 16
        CellLen = estimator.n_grid_points_y  # cell height, 16

        Total_out_m = np.empty((self.n_runs, CellWid* estimator.Nx, CellLen* estimator.Ny), dtype=self.True_maps[0][0].dtype)
        for usr_idx in range( estimator.n_walkers ):
            Cell_xy = index2Cell(usr_idx, estimator.Nx, estimator.Ny)
            usr_out_v = estimator.chosen_models[usr_idx].predict( #usr_out is (btsz,400,1)
                data_dict_pd[:,
                            Cell_xy[0] * CellWid: (Cell_xy[0] + 1) * CellWid + 2*ovp, #For input
                            Cell_xy[1] * CellLen: (Cell_xy[1] + 1) * CellLen + 2*ovp,
                             :],
            batch_size=self.btsz)
            usr_out_m = np.reshape(usr_out_v, ( self.n_runs, CellWid+2*ovp, CellLen+2*ovp)) + 0
            Total_out_m[:, Cell_xy[0] * CellWid: (Cell_xy[0] + 1) * CellWid,
                            Cell_xy[1] * CellLen: (Cell_xy[1] + 1) * CellLen] = usr_out_m[:, ovp:-ovp, ovp:-ovp] + 0

            print('User ', usr_idx, 'is tested')


        out = np.reshape(Total_out_m, self.TrueMapArray.shape)


        print(sampl_rate, 'sampling factor maps reconstructed!')
        rmse=(npla.norm((1-self.MetaMapArray)* (out - self.TrueMapArray)))**2 / np.sum(1-self.MetaMapArray)
        print(sampl_rate, 'sampling factor tested!')
        return rmse


class SimulatorNewJoint:
    """
        Combine centralized and FL RMSE testing
        Try to improve efficiency through batch processing (feed batch to DNN and calculate error in the similar way)
       Arguments:
    """

    def __init__(self,
                 n_runs=1,
                 use_parallel_proc=False,
                 generator=None,
                 sampler=None,
                 sampling_factors=None,
                 btsz = 512): #batchsize

        self.n_runs = n_runs
        self.use_parallel_proc = use_parallel_proc
        self.btsz = btsz
        self.generator = generator
        self.sampler = sampler
        self.sampling_factors = sampling_factors #list
        # ==== generate dataset for simulator with sampling_factor and n_runs each sampling rate ====
        self.data_dic = {a: None for a in sampling_factors } # prepare testing data of diff sampling rate, one big ndarray for sampling_rate, n_runsx48x48x3

        # 1. Loop to generate n_runs true_maps and save in a list. 2. Loop to sample len(sampling_factor) testsets from true_map
        # self.True_maps is a list containing n_runs tuples "( t_map 48x48x1, m_meta_map 48x48, __ )"
        num_cores = int(multiprocessing.cpu_count())
        self.True_maps = Parallel(n_jobs=num_cores, backend='threading')(delayed(self.generator.generate)() for _ in range(n_runs) )
        shap2 = self.True_maps[0][0].shape[2] #shap2=1, (48,48,1)
        print('True maps generated')
        self.TrueMapArray = np.array( #label(true map) for calculate RMSE, n_runs x 2304 x 1
            [np.expand_dims(TrueMap[0].flatten(), axis=1) for TrueMap in self.True_maps]) #flattened true maps in a list without mask/meta_map
            # ,axis=0)
        print('True maps flattened')

        # Save the flattened version of meta_map, the error in buildings is ignored.
        metamaplist = [np.expand_dims(np.repeat(true_map[1][:, :, np.newaxis], shap2, axis=2).flatten(), axis=1) for
                         true_map in self.True_maps]# list of (2304,1)
        self.MetaMapArray = np.array(metamaplist) # (n_runs, 2304, 1)
        print('Weights for loss calculation prepared')

        for a in sampling_factors: # handle the input data for each sampling factor
            self.sampler.v_sampling_factor = a
            # parallel sampler got list of tuples "(t_sampled_map_allfreq 48x48x1, mask 48x48)"
            sampled_a = Parallel(n_jobs=num_cores, backend='threading')(delayed(self.sampler.sample_map_customer)(self.True_maps[nn][0], self.True_maps[nn][1]) for nn in range(int(n_runs)) )

            self.data_dic[a] = np.empty((self.n_runs, 48,48,2 ), dtype=self.True_maps[0][0].dtype)

            for i, smlpd in enumerate(sampled_a): # save inputs of one sampling factor in an ndarray
                self.data_dic[a][i] = np.concatenate( # sensor_mask - meta_map, sharing a channel
                    (
                        np.expand_dims(smlpd[0],axis=0),
                        np.expand_dims(np.expand_dims(smlpd[1],axis=0), axis=3) - #sampled mask
                        np.expand_dims(np.expand_dims(-1*self.True_maps[i][1],axis=0), axis=3), # meta map
                        )
                        , axis=3
                )
            print( a, 'sampling factor data prepared.')

            # self.estimator.chosen_model.predict(x=self.data_dic[ ], batch_size=1024)
            # stack in to tensor
            # t_sampled_map_allfreq, mask = sampler.sample_map(t_map, m_meta_map)
        # for mapn in range(len(True_maps)): #sample map according to
        # l_l_data_points = Parallel(n_jobs=num_cores, backend='threading')(delayed(self.process_one_map)(ind_map=None) for ind_map in range(int(n_runs)))

    def simulateFL(self,
                 sampl_rate,
                 estimator,
                local = False,
                 ):
        """
        New version to simulate in batch, FL and standalone cases
        """
        #===== use DNN to process inputs in batches, save outputs =====
        # out = estimator.chosen_model.predict(x=self.data_dic[sampl_rate], batch_size=self.btsz)
        # For single user, zero-pad and cut the self.data_dic[sampl_rate]
        ovp = estimator.overlap
        pad_width = ((0,0),(ovp, ovp), (ovp, ovp), (0, 0))
        data_dict_pd = np.pad(self.data_dic[sampl_rate], pad_width=pad_width, mode='constant', constant_values=0)
        # Feed in local user and get vector output (btsz, 400,1)
        CellWid = estimator.n_grid_points_x  # cell width, 16
        CellLen = estimator.n_grid_points_y  # cell height, 16

        Total_out_m = np.empty((self.n_runs, CellWid* estimator.Nx, CellLen* estimator.Ny), dtype=self.True_maps[0][0].dtype)
        for usr_idx in range( estimator.n_walkers ):
            Cell_xy = index2Cell(usr_idx, estimator.Nx, estimator.Ny)
            if local:
                chosen = estimator.chosen_models[usr_idx]
            else:
                chosen = estimator.chosen_model
            usr_out_v = chosen.predict(  # usr_out is (btsz,400,1)
                        data_dict_pd[:,
                                    Cell_xy[0] * CellWid: (Cell_xy[0] + 1) * CellWid + 2*ovp, #For input
                                    Cell_xy[1] * CellLen: (Cell_xy[1] + 1) * CellLen + 2*ovp,
                                     :],
                    batch_size=self.btsz)
            usr_out_m = np.reshape(usr_out_v, ( self.n_runs, CellWid+2*ovp, CellLen+2*ovp)) + 0
            Total_out_m[:, Cell_xy[0] * CellWid: (Cell_xy[0] + 1) * CellWid,
                            Cell_xy[1] * CellLen: (Cell_xy[1] + 1) * CellLen] = usr_out_m[:, ovp:-ovp, ovp:-ovp] + 0

            print('User ', usr_idx, 'is tested')
        out = np.reshape(Total_out_m, self.TrueMapArray.shape)
        print(sampl_rate, 'sampling factor maps reconstructed!')
        rmse=(npla.norm((1-self.MetaMapArray)* (out - self.TrueMapArray)))**2 / np.sum(1-self.MetaMapArray)
        print(sampl_rate, 'sampling factor tested!')
        return rmse

    def simulateCent(self,
                 sampl_rate,
                 estimator,
                 ):
        """
        New version to simulate in batch, centralized case
        :param generator:  generator for generating maps used for training
        :type generator:   class
        :param sampler:   sampler for sampling the generated maps
        :type sampler:    class
        :param estimator:  estimator that reconstructs the sampled map
        :type estimator:
        :return:  the estimation error
        :type estimator:
        :return:real_map  consider_shadowing
        :rtype: float
        """
        sq_err_all_runs_arr = np.zeros((1, self.n_runs)) #initialize mse list
        #===== use DNN to process inputs in batches, save outputs =====
        #===== Calculate error (serial? together? in batch? TF parallel?)
        out = estimator.chosen_model.predict(x=self.data_dic[sampl_rate], batch_size=self.btsz)

        print(sampl_rate, 'sampling factor maps reconstructed!')
        rmse=(npla.norm((1-self.MetaMapArray)* (out - self.TrueMapArray)))**2 / np.sum(1-self.MetaMapArray)
            # sq_err_one_runs = (npla.norm((1 - v_meta) * (v_map - v_est_map))) ** 2 / len(
            #     np.where(v_meta == 0)[0])  # use [0] cuz the output is a tuple

            # self.MetaMapArray
        # rmse = np.sqrt(np.mean(sq_err_all_runs_arr))
        print(sampl_rate, 'sampling factor tested!')
        return rmse


class SimulatorSSC:
    """ for SSC data
       Arguments:
              n_runs: the number of monte carlo runs
    """
    def __init__(self,
                 n_runs=1,
                 use_parallel_proc=False):

        self.n_runs = n_runs
        self.use_parallel_proc = use_parallel_proc

    def simulate(self,
                 data_list,
                 sampler,
                 estimator,
                 ):
        """
        :param generator:  generator for generating maps used for training
        :type generator:   class
        :param sampler:   sampler for sampling the generated maps
        :type sampler:    class
        :param estimator:  estimator that reconstructs the sampled map
        :type estimator:
        :return:  the estimation error
        :type estimator:
        :return:real_map  consider_shadowing
        :rtype: float
        """
        def process_one_run(ind_run):
            # t_map, m_meta_map, _ = generator.generate()
            t_map = data_list[random.randint(0, len(data_list) - 1)].copy()[:, :, np.newaxis]
            m_meta_map = np.zeros((32, 32))

            m_meta_map_all_freqs = np.repeat(m_meta_map[:, :, np.newaxis], t_map.shape[2], axis=2)
            t_sampled_map_allfreq, mask = sampler.sample_map(t_map, m_meta_map)
            # start_time = time.time()
            t_estimated_map = estimator.estimate_map(t_sampled_map_allfreq, mask, m_meta_map)
            # end_time = time.time()
            # elapsed_time = end_time - start_time
            # print('The run-time of the estimator XX  is %.5f' % elapsed_time),
            v_meta = m_meta_map_all_freqs.flatten()
            v_map = t_map.flatten()
            v_est_map = t_estimated_map.flatten()
            sq_err_one_runs = (npla.norm((1 - v_meta) * (v_map - v_est_map))) ** 2 / len(
                np.where(v_meta == 0)[0])
            sq_err_one_runs_mw = npla.norm(  (1 - v_meta)*(10**(v_map/10)-10**(v_est_map/10)), ord=1) / len(np.where(v_meta == 0)[0])
            return sq_err_one_runs, sq_err_one_runs_mw

        if self.use_parallel_proc:
            num_cores = int(multiprocessing.cpu_count() / 2)
            sq_err_all_runs, sq_err_all_runs_mw = Parallel(n_jobs=num_cores)(delayed(process_one_run)(i)
                                                         for i in range(self.n_runs))
            sq_err_all_runs_arr = np.array(sq_err_all_runs)
            sq_err_all_runs_arr_mw = np.array(sq_err_all_runs_mw)
        else:
            sq_err_all_runs_arr = np.zeros((1, self.n_runs))
            sq_err_all_runs_arr_mw = np.zeros((1, self.n_runs))
            for ind_run in range(self.n_runs):
                sq_err_all_runs_arr[0, ind_run], sq_err_all_runs_arr_mw[0, ind_run] = process_one_run(ind_run)
        rmse = np.sqrt(np.mean(sq_err_all_runs_arr))
        rmse_mw = np.mean(sq_err_all_runs_arr_mw)
        return rmse, rmse_mw

class SimulatorSSC_Batch:
    """
       Arguments: Unfinished
              n_runs: the number of monte carlo runs
              work in batch to improve efficiency
    """
    def __init__(self,
                 n_runs=1,
                 use_parallel_proc=False):

        self.n_runs = n_runs
        self.use_parallel_proc = use_parallel_proc

    def simulate(self,
                 data_list,
                 sampler,
                 estimator,
                 ):
        """
        :param generator:  generator for generating maps used for training
        :type generator:   class
        :param sampler:   sampler for sampling the generated maps
        :type sampler:    class
        :param estimator:  estimator that reconstructs the sampled map
        :type estimator:
        :return:  the estimation error
        :type estimator:
        :return:real_map  consider_shadowing
        :rtype: float
        """
        def process_one_run(ind_run):
            # t_map, m_meta_map, _ = generator.generate()
            t_map = data_list[random.randint(0, len(data_list) - 1)].copy()[:, :, np.newaxis]
            m_meta_map = np.zeros((32, 32))

            m_meta_map_all_freqs = np.repeat(m_meta_map[:, :, np.newaxis], t_map.shape[2], axis=2)
            t_sampled_map_allfreq, mask = sampler.sample_map(t_map, m_meta_map)
            # start_time = time.time()
            t_estimated_map = estimator.estimate_map(t_sampled_map_allfreq, mask, m_meta_map)
            # end_time = time.time()
            # elapsed_time = end_time - start_time
            # print('The run-time of the estimator XX  is %.5f' % elapsed_time),
            v_meta = m_meta_map_all_freqs.flatten()
            v_map = t_map.flatten()
            v_est_map = t_estimated_map.flatten()
            sq_err_one_runs = (npla.norm((1 - v_meta) * (v_map - v_est_map))) ** 2 / len(
                np.where(v_meta == 0)[0])
            sq_err_one_runs_mw = npla.norm(  (1 - v_meta)*(10**(v_map/10)-10**(v_est_map/10)), ord=1) / len(np.where(v_meta == 0)[0])
            return sq_err_one_runs, sq_err_one_runs_mw

        if self.use_parallel_proc:
            num_cores = int(multiprocessing.cpu_count() / 2)
            sq_err_all_runs, sq_err_all_runs_mw = Parallel(n_jobs=num_cores)(delayed(process_one_run)(i)
                                                         for i in range(self.n_runs))
            sq_err_all_runs_arr = np.array(sq_err_all_runs)
            sq_err_all_runs_arr_mw = np.array(sq_err_all_runs_mw)
        else:
            sq_err_all_runs_arr = np.zeros((1, self.n_runs))
            sq_err_all_runs_arr_mw = np.zeros((1, self.n_runs))
            for ind_run in range(self.n_runs):
                sq_err_all_runs_arr[0, ind_run], sq_err_all_runs_arr_mw[0, ind_run] = process_one_run(ind_run)
        rmse = np.sqrt(np.mean(sq_err_all_runs_arr))
        rmse_mw = np.mean(sq_err_all_runs_arr_mw)
        return rmse, rmse_mw


class Simulator_FLOverlap:
    """
       Arguments:
              n_runs: the number of monte carlo runs

    """

    def __init__(self,
                 n_runs=1,
                 use_parallel_proc=False):

        self.n_runs = n_runs
        self.use_parallel_proc = use_parallel_proc

    def simulate(self,
                 generator,
                 sampler,
                 estimator,
                 ):
        """
        :param generator:  generator for generating maps used for training
        :type generator:   class
        :param sampler:   sampler for sampling the generated maps
        :type sampler:    class
        :param estimator:  estimator that reconstructs the sampled map
        :type estimator:
        :return:  the estimation error
        :type estimator:
        :return:real_map  consider_shadowing
        :rtype: float
        """
        def process_one_run(ind_run):
            t_map, m_meta_map, _ = generator.generate()
            m_meta_map_all_freqs = np.repeat(m_meta_map[:, :, np.newaxis], t_map.shape[2], axis=2)
            t_sampled_map_allfreq, mask = sampler.sample_map(t_map, m_meta_map)
            # start_time = time.time()
            t_estimated_map = estimator.estimate_map(t_sampled_map_allfreq, mask, m_meta_map)
            # end_time = time.time()
            # elapsed_time = end_time - start_time
            # print('The run-time of the estimator XX  is %.5f' % elapsed_time),
            v_meta = m_meta_map_all_freqs.flatten()
            v_map = t_map.flatten()
            v_est_map = t_estimated_map.flatten()
            sq_err_one_runs = (npla.norm((1 - v_meta) * (v_map - v_est_map))) ** 2 / len(
                np.where(v_meta == 0)[0])
            return sq_err_one_runs

        if self.use_parallel_proc:
            num_cores = int(multiprocessing.cpu_count() / 2)
            sq_err_all_runs = Parallel(n_jobs=num_cores)(delayed(process_one_run)(i)
                                                         for i in range(self.n_runs))
            sq_err_all_runs_arr = np.array(sq_err_all_runs)
        else:
            sq_err_all_runs_arr = np.zeros((1, self.n_runs))
            for ind_run in range(self.n_runs):
                sq_err_all_runs_arr[0, ind_run] = process_one_run(ind_run)
        rmse = np.sqrt(np.mean(sq_err_all_runs_arr))
        return rmse

def index2Cell( index, Nx, Ny):
    # For FL system working on a map partitioned into Nx x Ny cells,
    # Return the (x,y) cell directory for FL worker-'index'
    # Input range: For 9 cell: index in [0, 8], Nx=Ny=3. Output: row, col in [0,2]
    if index > (Nx *Ny-1):
        print('invalid worker index')
    index =index +1 # convenience
    row = (index -1) //Ny +1
    col = (index -1) % Nx +1
    return (row - 1, col - 1)