import os
from utilities.utils import write_print, mkdir
import argparse
from solver import Solver
from data.data_loader import get_loader
from torch.backends import cudnn
from datetime import datetime
import zipfile
import torch
import numpy as np


def zip_directory(path, zip_file):
    """Stores all py and cfg project files inside a zip file

    Arguments:
        path {string} -- current path
        zip_file {zipfile.ZipFile} -- zip file to contain the project files
    """
    files = os.listdir(path)
    for file in files:
        if file.endswith('.py') or file.endswith('cfg'):
            zip_file.write(os.path.join(path, file))
            if file.endswith('cfg'):
                os.remove(file)


def save_config(path, version, config):
    """saves the configuration of the experiment

    Arguments:
        path {str} -- save path
        version {str} -- version of the model based on the time
        config {dict} -- contains argument and its value

    """
    cfg_name = '{}.{}'.format(version, 'cfg')

    with open(cfg_name, 'w') as f:
        for k, v in config.items():
            f.write('{}: {}\n'.format(str(k), str(v)))

    zip_name = '{}.{}'.format(version, 'zip')
    zip_name = os.path.join(path, zip_name)
    zip_file = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    zip_directory('.', zip_file)
    zip_file.close()


def string_to_boolean(v):
    """Converts string to boolean

    Arguments:
        v {string} -- string representation of a boolean values;
        must be true or false

    Returns:
        boolean -- boolean true or false
    """
    return v.lower() in ('true')


def main(version, config, output_txt, compile_txt):
    """Runs either Solver or Compressor object

    Arguments:
        v {string} -- string representation of a boolean values;
        must be true or false

    Returns:
        config {dict} -- contains argument and its value
        output_txt {str} -- file name for the text file where details are logged
        compile_txt {str} -- file name for the text file where performance is compiled (if val/test mode)
    """

    # for fast training
    cudnn.benchmark = True


    # if config.use_compress:
    #     config.mode = 'train'
    #     train_loader, _ = get_loader(config)

    #     if config.dataset == 'micc':
    #         config.mode = 'val'
    #     else:
    #         config.mode = 'test'
    #     val_loader, dataset_ids = get_loader(config)

    #     data_loaders = {
    #         'train': train_loader,
    #         'val': val_loader
    #     }
    #     compressor = Compressor(data_loaders, dataset_ids, vars(config), output_txt)
    #     compressor.compress()
    #     return

    data_loader, dataset_ids = get_loader(config)
    print(data_loader)
    solver = Solver(version, data_loader, dataset_ids, vars(config), output_txt, compile_txt)

    if config.mode == 'train':
        temp_save_path = os.path.join(config.model_save_path, version)
        mkdir(temp_save_path)
        solver.train()

    elif config.mode == 'val' or config.mode == 'test':
        solver.test()

    elif config.mode == 'pred':
        solver.pred()


if __name__ == '__main__':
    torch.set_printoptions(threshold=np.nan)
    parser = argparse.ArgumentParser()

    # dataset info
    parser.add_argument('--dataset', type=str, default='shanghaitech-a',
                        choices=['shanghaitech-a', 'shanghaitech-b', 'ucf-cc-50', 'ucf-qnrf'],
                        help='Dataset to use')
    
    # training settings
    parser.add_argument('--lr', type=float, default=1e-7,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.95,
                        help='Momentum')
    parser.add_argument('--weight_decay', type=float, default= 0.0001,
                        help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--learning_sched', type=list, default=[],
                        help='List of epochs to reduce the learning rate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--model', type=str, default='CSRNet',
                        choices=['CSRNet', 'CAN', 'MAN', 'ConNet'],
                        help='CNN model to use')
    parser.add_argument('--pretrained_model', type=str,
                        default='C:/Users/lande/Desktop/THS-ST2/Pipeline/weights/CSRNet shanghaitech-a 2023-01-10 17_35_58.055399_train/30.pth',
                        help='Pre-trained model')
    parser.add_argument('--save_output_plots', type=string_to_boolean, default=True)
    parser.add_argument('--init_weights', type=string_to_boolean, default=True,
                        help='Toggles weight initialization')
    # parser.add_argument('--fail_cases', type=string_to_boolean, default=False,
    #                     help='Toggles identification of failure cases')

    # misc
    parser.add_argument('--mode', type=str, default='pred',
                        choices=['train', 'val', 'test', 'pred'],
                        help='Mode of execution')
    parser.add_argument('--use_gpu', type=string_to_boolean, default=True,
                        help='Toggles the use of GPU')

    # epoch step size
    parser.add_argument('--loss_log_step', type=int, default=5)
    parser.add_argument('--model_save_step', type=int, default=5)

    # for experiments
    # parser.add_argument('--augment_exp', type=string_to_boolean, default=False)
    # parser.add_argument('--brightness_change', type=float, default=0)
    # parser.add_argument('--resolution_scale', type=float, default=1)
    # parser.add_argument('--outdoor', type=string_to_boolean, default=False)

    ############# FILE PATHS #############
    parser.add_argument('--model_save_path', type=str, default='./weights',
                        help='Path for saving weights')
    parser.add_argument('--model_test_path', type=str, default='/tests',
                        help='Path for saving test results')

    # ShanghaiTechA dataset
    parser.add_argument('--shanghaitech_a_path', type=str,
                        default='../Datasets/ShanghaiTechA/',
                        help='ShanghaiTech A dataset path')
    # ShanghaiTechB dataset
    parser.add_argument('--shanghaitech_b_path', type=str,
                        default='../Datasets/ShanghaiTechB/',
                        help='ShanghaiTech B dataset path')
    # UCF_CC_50 dataset
    parser.add_argument('--ucf_cc_50_path', type=str,
                        default='../Datasets/UCF-CC-50/',
                        help='UCF-CC-50 dataset path')
    # UCF_QNRF dataset
    parser.add_argument('--ucf_qnrf_path', type=str,
                        default='../Datasets/UCF-QNRF/',
                        help='UCF-QNRF dataset path')


    config = parser.parse_args()

    args = vars(config)
    output_txt = ''

    # Preparation of details for Solver object (if use_compress == False)
    if args['mode'] == 'train':
        dataset = args['dataset']
       

        version = str(datetime.now()).replace(':', '_')
        # version = '{} {} bright_{} res_{} {}_train'.format(args['model'], dataset, args['brightness_change'], args['resolution_scale'], version)

        version = '{} {} {}_train'.format(args['model'], dataset, version)
        path = args['model_save_path']
        path = os.path.join(path, version)
        output_txt = os.path.join(path, '{}.txt'.format(version))
        compile_txt = os.path.join(path, 'COMPILED {} {}.txt'.format(args['model'], version))

    elif args['mode'] == 'val':
        # C:/Users/lande/Desktop/THS-ST2/Pipeline/weights/CSRNet shanghaitech-a 2023-01-09 16_51_40.548283_train/10.pth
        model = args['pretrained_model'].split('/')
        version = '{}_test_{}'.format(model[-2], model[-1])
        # pretrained/12-21-2022/2
        # args['model_test_path'] += '/' + '/'.join(model[:-1])
        path = '/'.join(model[:-3]) + args['model_test_path']
        # path = os.path.join(path, model[0])
        output_txt = os.path.join(path, '{}.txt'.format(version))
        compile_txt = os.path.join(path, 'COMPILED {} {} {}.txt'.format(args['model'], args['mode'], model[0]))

    elif args['mode'] == 'test':
        model = args['pretrained_model'].split('/')
        version = '{}_test_{}'.format(model[-2], model[-1])
        
        # args['model_test_path'] += '/' + '/'.join(model[:-1])
        path = '/'.join(model[:-3]) + args['model_test_path']
        # path = os.path.join(path, model[0])
        output_txt = os.path.join(path, '{}.txt'.format(version))
        compile_txt = os.path.join(path, 'COMPILED {} {} {}.txt'.format(args['model'], args['mode'], model[0]))

    elif args['mode'] == 'pred':
        model = args['pretrained_model'].split('/')
        version = '{}_test_{}'.format(model[-2], model[-1])
        
        # args['model_test_path'] += '/' + '/'.join(model[:-1])
        path = '/'.join(model[:-3]) + args['model_test_path']
        # path = args['model_test_path']
        path = os.path.join(path, model[0])
        output_txt = os.path.join(path, '{}.txt'.format(version))
        compile_txt = os.path.join(path, 'COMPILED {} {}.txt'.format(args['model'], model[0]))

    # create folder and save copy of files
    mkdir(path)
    save_config(path, version, args)

    # log the settings in output file
    write_print(output_txt, '------------ Options -------------')
    for k, v in args.items():
        write_print(output_txt, '{}: {}'.format(str(k), str(v)))
    write_print(output_txt, '-------------- End ----------------')

    main(version, config, output_txt, compile_txt)
