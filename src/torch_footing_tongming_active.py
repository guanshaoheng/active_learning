import time

from FEMxML.torch_active_learning_onthefly import main_active_def
from FEMxML.utils_ml import echo, get_data_series, data_clean_numg
from utilSelf.general import check_mkdir


def main(
        data_index=0,
        max_time=70,
        outer_directory='./footing_ml/active_footing_3618',
        fourier_features=False,
        epoch_per_iter=int(2e4),
        series_flag=True,
):
    check_mkdir(outer_directory)
    echo('\tReading data ...')
    data_paths = [
        '../../simu/footing/footing_dem_footing_1206_2D_order2_numG3618_2ndShear',
    ]
    data_paths = [data_paths[data_index]]
    numg = int(data_paths[data_index].split('numG')[1].split('_')[0])
    # datas_dict = get_data(
    #     root_path_list=data_paths, maxTime=int(max_time), series_flag=False)
    datas_dict = get_data_series(root_path_list=data_paths, maxTime=int(max_time), series_flag=series_flag, numg=numg, )
    datas_dict = data_clean_numg(datas_dict)

    main_active_def(
        out_directory=outer_directory,
        datas=datas_dict,
        input_features='epsANDabsxy',
        output_features='sig',
        node=5, layerList='dd', fourier_features=fourier_features,
        iter_max=4, ratio_per_iter=0.02, epoch_per_iter=epoch_per_iter,
        first_train_ratio=0.02, first_epoch_num=epoch_per_iter,
        remove_used_sample_flag=False,
    )

    main_active_def(
        out_directory=outer_directory,
        datas=datas_dict,
        input_features='epsANDabsxy',
        output_features='D',
        node=8, layerList='dd', fourier_features=fourier_features,
        iter_max=4, ratio_per_iter=0.02, epoch_per_iter=epoch_per_iter,
        first_train_ratio=0.02, first_epoch_num=epoch_per_iter,
        remove_used_sample_flag=False,
    )


if __name__ == '__main__':
    start_time = time.time()
    main()
    echo('Total time consumed: %.2e mins' % ((time.time() - start_time) / 60.))
