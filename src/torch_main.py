import os

import numpy as np

from FEMxML.torch_net import Net
from FEMxML.torch_restore import modelRestore
from FEMxML.torch_training import train_class, train_dy_class
from FEMxML.utils_ml import \
    get_data, findDevice, pickle_load, check_mkdir, save_scalar, reconstruct_x_y, sampling_index
from utilSelf.general import echo


class MainTrain:
    def __init__(self, savePath, dy_flag=False, inputNum=4, outputNum=1, layerList='dd',
                 fourier_features=False, node_num=30, dy_weight=1.0):
        self.savedPath = savePath
        self.dy_flag = dy_flag
        self.fourier_features = fourier_features
        echo('Working directory: %s' % self.savedPath)
        self.device = findDevice()
        self.input_mean, self.input_std, self.output_mean, self.output_std = pickle_load(
            'input_mean', 'input_std', 'output_mean', 'output_std',
            root_path=self.savedPath)
        self.inputNum = inputNum
        self.outputNum = outputNum
        self.layerList = layerList
        self.node_num = node_num
        self.dy_weight = dy_weight
        # initial the trainer.
        self.trainer = train_class(
            model=self.get_initial_network_model(), patienceNum=int(5e3), savePath=self.savedPath, optimMode='adam',
                                  device=self.device)

    def get_initial_network_model(self):
        echo('initial model')
        return Net(
            inputNum=self.inputNum, outputNum=self.outputNum,
            xmean=self.input_mean, xstd=self.input_std, ymean=self.output_mean, ystd=self.output_std,
            layerList=self.layerList,
            fourier_features=self.fourier_features, node=self.node_num, device=self.device)

    def initial_network_model_in_trainer(self):
        """
            Caution: redefine the model AND THE OPTIMIZER
        """
        self.trainer.model = self.get_initial_network_model()
        self.trainer.optimode_selection(optimMode='adam')

    def restore_evaluation(self):
        model_restored = modelRestore(savedPath=self.savedPath)
        return model_restored

    def evaluation(self, x_data, y_data):
        model_restored = self.restore_evaluation()
        np.random.seed(10002)
        index = sampling_index(x_data, sample_num=1000)
        model_restored.modelEvaluation(inputs=x_data[index], outputs=y_data[index])
        return

    def train(self, x, y, epoch_max=None, optimMode='adam'):
        self.trainer.train(inputs=x, outputs=y, optimMode=optimMode, epochMax=epoch_max)
        return


class MainTrain_dy(MainTrain):
    def __init__(self, savePath, input_num, output_num,
                 fourier_features, layerList, dy_weight, node_num):
        MainTrain.__init__(
            self, savePath=savePath, dy_flag=True, dy_weight=dy_weight, inputNum=input_num,
            outputNum=output_num, fourier_features=fourier_features, layerList=layerList, node_num=node_num)
        self.trainer = train_dy_class(
            model=self.get_initial_network_model(), patienceNum=int(5e3), savePath=self.savedPath,
            device=self.device,
            optimMode='adam', dyWeight=self.dy_weight, save_model_during_process_flag=False,
            num_batches=-1, verbose_flag=False)

    def train(self, x, y, dy, epoch_max=None, optimMode='adam'):
        self.trainer.train(x=x, y=y, dy=dy, optimMode=optimMode, epoch_max=epoch_max)
        return

    def evaluation(self, x, y, dy):
        model_restored = self.restore_evaluation()
        np.random.seed(10002)
        index = sampling_index(x, sample_num=1000)
        model_restored.modelEvaluation_y_dy(x=x[index], y=y[index], dy=dy[index])





def train_main_def(
        datas,
        outer_directory,
        input_features='epsANDabsxy', output_features='D', rotate_flag=False,
        layerList='dd',
        node_num=20,
        fourier_features=False, epoch_max=int(1e5), numSamplesUsed = int(2e5),
        special_str=None, sample_ratio=None):
    dir_name = 'X_%s_Y_%s_%s%d_%s_%s' % (
        input_features, output_features, layerList, node_num,
        ('Fourier' if fourier_features else 'noFourier'),
        ('rotate' if rotate_flag else 'noRotate'))
    if special_str and special_str != '':
        dir_name += '_%s' % special_str
    save_path = os.path.join(
        outer_directory, dir_name)
    echo(save_path)
    check_mkdir(save_path)
    x_data, y_data = reconstruct_x_y(
        input_features=input_features, output_features=output_features, rotate_flag=rotate_flag, **datas)
    save_scalar(scalarPath=save_path, input_value=x_data, output_value=y_data)
    temp = MainTrain(
        savePath=save_path, inputNum=len(x_data[0]), outputNum=len(y_data[0]),
        node_num=node_num, layerList=layerList,
        fourier_features=fourier_features)
    index = sampling_index(x_data, sample_num=numSamplesUsed, ratio=sample_ratio)
    temp.train(x_data[index], y_data[index], epoch_max=epoch_max)
    temp.evaluation(x_data, y_data)


def train_main_dy_mask(
        x, y, dy, dy_weight, node_num,
        outer_directory, numSamplesUsed=int(1e4), epoch_max=int(1e4), fourier_features=False, layerList='dddd'):
    save_path = os.path.join(outer_directory, 'vonmises_%s_%s%d_dyweight%.1f' %
                             (('Fourier' if fourier_features else 'noFourier'), layerList, node_num, dy_weight))
    echo(save_path)
    check_mkdir(outer_directory, save_path)
    save_scalar(scalarPath=save_path, input_value=x, output_value=y)
    temp = MainTrain_dy(savePath=save_path, input_num=len(x[0]), output_num=len(y[0]),
                        fourier_features=fourier_features, layerList=layerList,
                        dy_weight=dy_weight, node_num=node_num)
    index = sampling_index(x, sample_num=numSamplesUsed)
    temp.train(x=x[index], y=y[index], dy=dy[index], epoch_max=epoch_max)
    temp.evaluation(x, y, dy)


if __name__ == '__main__':
    outer_directory = './footing_ml'
    check_mkdir(outer_directory)

    echo('\tReading data ...')
    data_paths = ['../../simu/footing/footing_dem_footing552_2D_order1_numG480']
    returned_dic = get_data(
        root_path_list=data_paths, maxTime=int(60))

    # ------------------------- D ---------------------------
    train_main_def(
        datas=returned_dic,
        input_features='epsANDabsxy', output_features='D',
        layerList='dd',
        node_num=20,
        fourier_features=False, outer_directory=outer_directory)

    # ------------------------- sig ---------------------------
    train_main_def(
        datas=returned_dic,
        input_features='epsANDabsxy', output_features='sig',
        layerList='dd',
        node_num=20,
        fourier_features=False, outer_directory=outer_directory)
