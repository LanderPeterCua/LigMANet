import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # make into list if more than one GPU

import numpy as np
from torch.backends import cudnn
import torch

from models.CSRNet.CSRNetSolver import CSRNetSolver
from models.CSRNet.CSRNetDataset import CSRNetDataset

from models.CAN.CANSolver import CANSolver
from models.CAN.CANSolverSKT import CANSolverSKT
from models.CAN.CANSolverPruned import CANSolverPruned

from models.MAN.MANSolver import MANSolver
from models.MAN.MANSolverSKT import MANSolverSKT
from models.MAN.MANSolverPruned import MANSolverPruned
import argparse


# TODO Add parameters to the constructor during THS-ST3 so that users can utilize the pipeline without changing the code
class Paths(object):
    """ A class that contains all the paths to be used in the pipeline

    Attributes:
        pretrained_model (string): path to the pretrained model for validation, test, or prediction;
                                   set to None during training
        weights (string): path to the folder containing saved models
        test_results (string): path to the folder containing the results of testing
        shanghaitech_a (string): path to the folder containing the shanghaitech_a dataset
        shanghaitech_b (string): path to the folder containing the shanghaitech_b dataset
        ucf_cc_50 (string): path to the folder containing the ucf_cc_50 dataset
        ucf_qnrf (string): path to the folder containing the ucf_qnrf dataset
    """
    def __init__(self):
        self.pretrained_model = None 
        self.weights = './weights'
        self.test_results = './tests'
        self.shanghaitech_a = '../Datasets/ShanghaiTechA/'
        self.shanghaitech_b = '../Datasets/ShanghaiTechB/'
        self.ucf_cc_50 = '../Datasets/UCF-CC-50/folds/'
        self.ucf_qnrf = '../Datasets/UCF-QNRF/'
        self.man_shanghaitech_a = '../Datasets/ShanghaiTechAPreprocessed/'
        self.man_shanghaitech_b = '../Datasets/ShanghaiTechBPreprocessed/'
        self.man_ucf_cc_50 = '../Datasets/UCF-CC-50Preprocessed/folds/'
        self.man_ucf_qnrf = '../Datasets/UCF-QNRFPreprocessed/'

# TODO Add parameters to the constructor during THS-ST3 so that users can utilize the pipeline without changing the code
class Config(object):
    """ A class that contains all the needed configurations regarding the training/validation/prediction session.

    Attributes:
        mode (string) [Train, Val, Test, Pred]: determines how the model will be used within the pipeline
        model (string) [CSRNet, CAN, MAN, ConNet]: determines what crowd counting model to be used for the session
        dataset (string) [ShanghaiTech-A, ShanghaiTech-B, UCF-CC-50, UCF-QNRF]: determines what dataset the crowd counting model will be used on
        learning_rate (double): learning rate of the model
        momentum (double): momentum of the model
        weight_decay (double): weight decay of the model
        num_epochs (int): number of epochs for training
        batch_size (int): batch size of the model
        use_gpu (bool): True if the session will use the GPU, False if the session will use the CPU
    """
    def __init__(self):
        self.mode = "Train"                 
        self.model = "MAN"              
        self.dataset = "UCFCC50" # [Shanghaitech-A, Shanghaitech-B, UCFCC50, UCFQNRF] 
        self.cc50_val = 3 # [1, 2, 3, 4, 5]
        self.cc50_test = 5 # [1, 2, 3, 4, 5]
        
        #VGG
        self.lr = 5e-6
        self.learning_sched = []
        
        self.momentum = 0.95             
        self.weight_decay = 1e-5  
        self.num_epochs = 1200            
        self.batch_size = 1    
        self.use_gpu = True
        self.weights = "weights/VGG19-ShanghaiTech-A/best_model_9.pth"

        self.compression = True
        self.compression_technique = "SKT" # [pruning, SKT]
        self.lamb_fsp = None
        self.lamb_cos = None
        self.SKT_teacher_ckpt = "weights/0410-174037/916_teacher_ckpt.tar"
        self.SKT_student_ckpt = "weights/0410-174037/916_student_ckpt.tar"
        
        print('GPU:', torch.cuda.current_device())
        print('GPU Name:', torch.cuda.get_device_name(torch.cuda.current_device()))
        

def main():
    # for faster training
    cudnn.benchmark = True
    config = Config()
    paths = Paths()

    if (config.compression == False):
        if config.model == "CSRNet":
            solver = CSRNetSolver(config, paths)
            solver.start(config)

        elif config.model == "CAN":
            solver = CANSolver(config, paths)
            solver.start(config)

        elif config.model == "MAN":
            args = parse_args(config, paths)
            solver = MANSolver(args)
            
            if config.mode == "Train":
                solver.setup()
                solver.train()
            elif config.mode == "Test":
                solver.test(args)
    
    else:
        if config.compression_technique == "Pruning":
            if config.model == "CAN":
                solver = CANSolverPruned(config, paths)
                solver.start(config)
            
            elif config.model == "MAN":
                args = parse_args(config, paths)
                solver = MANSolverPruned(args)

                if config.mode == "Train":
                    solver.setup()
                    solver.train()
                elif config.mode == "Test":
                    solver.test(args)
        
        else:
            if config.model == "CAN":
                solver = CANSolverSKT(config, paths)
                solver.start(config)
            
            elif config.model == "MAN":
                args = parse_args(config, paths)
                solver = MANSolverSKT(args)

                if config.mode == "Train":
                    solver.setup()
                    solver.train()
                elif config.mode == "Test":
                    solver.test(args)

def parse_args(config, paths):
        config = config
        paths = paths
        
        if (config.dataset == "Shanghaitech-A"):
            dataset_path = paths.man_shanghaitech_a
        elif (config.dataset == "Shanghaitech-B"):
            dataset_path = paths.man_shanghaitech_b
        elif (config.dataset == "UCFCC50"):
              dataset_path = paths.man_ucf_cc_50
        else:
              dataset_path = paths.man_ucf_qnrf
              
        parser = argparse.ArgumentParser(description=config.mode)
        parser.add_argument('--model-name', default=config.model, help='the name of the model')
        parser.add_argument('--dataset-name', default=config.dataset, help='the name of the dataset')
        parser.add_argument('--data-dir', default=dataset_path,
                            help='training data directory')
        parser.add_argument('--cc-50-val', default=config.cc50_val, help='fold number to use as validation set for cc50')
        parser.add_argument('--cc-50-test', default=config.cc50_test, help='fold number to use as test set for cc50')
        parser.add_argument('--save-dir', default=paths.weights,
                            help='directory to save models.')
        parser.add_argument('--save-all', type=bool, default=True,
                            help='whether to save all best model')
        parser.add_argument('--lr', type=float, default=config.lr,
                            help='the initial learning rate')
        parser.add_argument('--weight-decay', type=float, default=config.weight_decay,
                            help='the weight decay')
        parser.add_argument('--resume', default=None,
                            help='the path of resume training model')
        parser.add_argument('--max-model-num', type=int, default=2,
                            help='max models num to save ')
        parser.add_argument('--max-epoch', type=int, default=config.num_epochs,
                            help='max training epoch')
        parser.add_argument('--val-epoch', type=int, default=1,
                            help='the num of steps to log training information')
        parser.add_argument('--val-start', type=int, default=600,
                            help='the epoch start to val')
        parser.add_argument('--batch-size', type=int, default=config.batch_size,
                            help='train batch size')
        parser.add_argument('--device', default='1', help='assign device')
        parser.add_argument('--num-workers', type=int, default=8,
                            help='the num of training process')

        parser.add_argument('--is-gray', type=bool, default=False,
                            help='whether the input image is gray')
        parser.add_argument('--crop-size', type=int, default=256,
                            help='the crop size of the train image')
        parser.add_argument('--downsample-ratio', type=int, default=16,
                            help='downsample ratio')
        parser.add_argument('--augment-contrast', type=bool, default=False,
                            help='whether to apply contrast enhancement on images')
        parser.add_argument('--augment-contrast-factor', type=float, default=1,
                            help='Contrast enhancment factor')

        parser.add_argument('--use-background', type=bool, default=True,
                            help='whether to use background modelling')
        parser.add_argument('--sigma', type=float, default=8.0,
                            help='sigma for likelihood')
        parser.add_argument('--background-ratio', type=float, default=0.15,
                            help='background ratio')
  
        parser.add_argument('--best-model-path', default=config.weights,
                            help='best model path')
        parser.add_argument('--learning-sched', default = config.learning_sched,
                            help='number of epochs for warmup learning')
        
        parser.add_argument('--compression', default = config.compression,
                            help='whether compression is to be implemented')
        parser.add_argument('--compression-technique', default = config.compression_technique,
                            help='compression technique to be used')
        parser.add_argument('--lamb-fsp', default = config.lamb_fsp,
                            help='weight of dense fsp loss')
        parser.add_argument('--lamb-cos', default = config.lamb_fsp,
                            help='weight of cos loss')
        parser.add_argument('--teacher_ckpt', default = config.SKT_teacher_ckpt,
                            help='SKT teacher checkpoint')
        parser.add_argument('--student_ckpt', default = config.SKT_student_ckpt,
                            help='SKT student checkpoint')
    
        args = parser.parse_args()
        return args

if __name__ == '__main__':
    main()
    