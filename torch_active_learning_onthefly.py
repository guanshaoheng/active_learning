import os
import time

import numpy as np
import torch

from FEMxML.utils_ml import save_scalar, reconstruct_x_y, echo, check_mkdir, findDevice, writeLine
from torch_main import MainTrain


class ActiveLearning:
    def __init__(self, datas, input_features, output_features, numNN,
                 node_num=8, fourier_features=False, out_dir='biax_ml_1e5',
                 layerList='dmdd'):
        # =================
        self.start_time = time.time()
        self.node_num = node_num
        self.fourier_features = fourier_features
        self.layerList = layerList
        self.device = findDevice()
        self.numNN = numNN
        self.save_path = '%s/X_%s_Y_%s_numNN%d_%s%d' % \
                         (out_dir, input_features, output_features, numNN, layerList, node_num)
        check_mkdir(self.save_path)
        self.datas = datas
        self.x_numg, self.y_numg = reconstruct_x_y(
            input_features=input_features, output_features=output_features, **self.datas)
        self.input_num, self.output_num = len(self.x_numg[0, 0]), len(self.y_numg[0, 0])
        self.x, self.y = self.x_numg.reshape([-1, self.input_num]), self.y_numg.reshape([-1, self.output_num])
        self.main_trains = self.getTrainer()
        self.total_numg = len(self.x_numg)
        self.total_sample_num = len(self.x)
        self.pool_left_index_numg = list(range(self.total_numg))
        self.pool_used_index_numg = []
        self.pool_left_index = list(range(self.total_sample_num))
        self.pool_used_index = []
        self.history_of_variance = []
        self.fname = os.path.join(self.save_path, 'sample_index.txt')
        writeLine(fname=self.fname, s='Variance & Used index during the training and resampling process\n', mode='w')
        writeLine(fname=self.fname, s='Total num %d \n' % self.total_sample_num, mode='a')

    def getTrainer(self, ):
        main_trains = []
        for i in range(self.numNN):
            save_path = os.path.join(self.save_path, 'active_%d' % i)
            echo(save_path)
            check_mkdir(save_path)
            save_scalar(scalarPath=save_path, input_value=self.x, output_value=self.y)
            main_trains.append(
                MainTrain(
                    savePath=save_path, inputNum=self.input_num, outputNum=self.output_num,
                          node_num=self.node_num, layerList=self.layerList,
                          fourier_features=False))
        return main_trains

    def train(self, index, epoch=None, iter=0,
              remove_used_sample_flag=False, numg_flag=True, intial_model_flag=True):
        if numg_flag:
            if remove_used_sample_flag:
                self.pool_left_index_numg = list(set(self.pool_left_index_numg) - set(index))
            self.pool_used_index_numg += list(index)
        else:
            if remove_used_sample_flag:
                self.pool_left_index = list(set(self.pool_left_index) - set(index))
            self.pool_used_index += list(index)

        line = 'Iter %d Used/Total gauss num: %d/%d\n' % (iter, len(self.pool_used_index_numg), self.total_numg)
        echo(line)
        echo("Added guass points number: [%s]" % ' '.join(['%d' % i for i in index]))
        writeLine(fname=self.fname, s=line, mode='a')
        writeLine(fname=self.fname,
                  s='Iter %d added index: [%s]\n' %\
                    (iter, ' '.join(['%d' % i for i in index])), mode='a')
        for i in range(self.numNN):
            echo(line, 'Traing of net %d' % i)
            if intial_model_flag:
                self.main_trains[i].initial_network_model_in_trainer()
            self.main_trains[i].train(
                x=self.x_numg[self.pool_used_index_numg].reshape([-1, self.input_num]),
                y=self.y_numg[self.pool_used_index_numg].reshape([-1, self.output_num]),
                epoch_max=epoch if epoch else int(2e4))
            self.main_trains[i].trainer.save(savedPath=self.main_trains[i].trainer.savePath, iter=iter)
            # only run the first network training if no index of training samples added
            if len(index) == 0:
                break
        return

    def resampling(self, ratio=None, numg=None, iter=0):
        predictions_numg = []
        for i in range(self.numNN):
            predictions_temp = []
            for numg_temp in self.pool_left_index_numg:
                predictions_temp.append(
                    self.main_trains[i].trainer.model(
                        torch.from_numpy(
                            self.x_numg[numg_temp]
                        ).float().to(self.device)).cpu().detach().numpy())
            predictions_numg.append(predictions_temp)

        predictions_numg = np.array(predictions_numg)  # [num_NN, numg, numg_step, n_features]
        std_prediction = np.std(predictions_numg, axis=0)  # [numg, numg_step, n_features]
        variance = np.mean(np.mean(std_prediction, axis=-1), axis=-1)  # [numg]
        if ratio:
            num_sampled = int(self.total_numg * ratio)
        elif numg:
            num_sampled = numg
        else:
            echo('ratio %.3e numg %.3e' % (ratio, numg))
            raise
        index_variance = np.argsort(variance)[::-1][int(0.1*num_sampled):num_sampled]
        index_numg = np.array(self.pool_left_index_numg)[index_variance]
        mean_variance = np.average(variance[index_variance])
        self.history_of_variance.append(mean_variance)
        s = '\n' + '=' * 80 + \
            '\nIter %d Mean variance of the highest %.3e of left %d/%d samples: %.3e used_num: %d\n' % (
                iter, ratio, num_sampled, len(self.pool_left_index_numg), mean_variance,
                len(self.pool_used_index_numg) + num_sampled)
        echo(s)
        writeLine(fname=self.fname, s=s, mode='a')
        return index_numg

    def evaluation(self):
        index = np.random.permutation(range(self.total_sample_num))[:1000]
        for i in range(self.numNN):
            self.main_trains[i].evaluation(x_data=self.x[index], y_data=self.y[index])


def main_active_def(
        datas,
        input_features='epsANDH', output_features='sigANDH', node=20,
        numNN=3, iter_max=5, ratio_per_iter=0.01, first_train_ratio=0.01, first_epoch_num=int(1e5),
        epoch_per_iter=int(2e4), layerList='dmd', fourier_features=False, out_directory=None,
        remove_used_sample_flag=False,
):
    # define the active learning class
    active_learning = ActiveLearning(
        datas=datas, input_features=input_features, output_features=output_features, numNN=numNN,
        node_num=node, fourier_features=fourier_features,
        layerList=layerList,
        out_dir=out_directory,)

    total_numg = active_learning.total_numg
    total_num_sample = active_learning.total_sample_num
    echo(
        'Total number of gauss points: %d' % total_numg,
        'Total number of samples     : %d' % total_num_sample,
         )

    # initial training (use some randomly picked gauss points)
    # index_initial = np.random.permutation(range(total_num_sample))[:int(first_train_ratio * total_num_sample)]
    index_initial_numg = np.random.permutation(range(total_numg))[:int(first_train_ratio * total_numg)]
    active_learning.train(
        index=index_initial_numg,
        epoch=first_epoch_num, iter=0,
        remove_used_sample_flag=remove_used_sample_flag, numg_flag=True)

    # active learning based resampling and retraining
    for a_num in range(1, iter_max + 1):
        index_resampling = active_learning.resampling(ratio=ratio_per_iter, iter=a_num)
        active_learning.train(
            index=index_resampling,
            epoch=epoch_per_iter, iter=a_num,
            remove_used_sample_flag=remove_used_sample_flag, numg_flag=True)
    echo('The variance history: %s' % active_learning.history_of_variance)

    # final train the indexed 0 Network
    active_learning.train(index=[], iter=iter_max+1, epoch=int(1e5), remove_used_sample_flag=remove_used_sample_flag)
    active_learning.evaluation()

    writeLine(
        fname=active_learning.fname,
        s='Total time consumed: %.2e mins' % ((time.time()-active_learning.start_time)/60.), mode='a')
