import os
import torch
import time
import copy
import logging
import datetime
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.model import get_model
from utilities.utils import to_var, write_print, write_to_file, save_plots, get_amp_gt_by_value
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, mean_squared_error, mean_absolute_error)
from utilities.timer import Timer
import numpy as np
import torch.nn.functional as F
from utilities.marunet_losses import cal_avg_ms_ssim
from losses.post_prob import Post_Prob
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from models.MAN import vgg_c
from losses.bay_loss import Bay_Loss
from utilities.helper import Save_Handle
from utilities.helper import AverageMeter

class Solver(object):

    DEFAULTS = {}

    def __init__(self, version, data_loader, dataset_ids, config, output_txt, compile_txt):
        """
        Initializes a Solver object

        Arguments:
            version {str} -- version of the model based on the time
            data_loader {DataLoader} -- DataLoader of the dataset to be used
            dataset_ids {list} -- list of image IDs, used for naming the exported density maps
            config {dict} -- contains arguments and its values
            output_txt {str} -- file name for the text file where details are logged
            compile_txt {str} -- file name for the text file where performance is compiled (if val/test mode)
        """

        # data loader
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.version = version
        self.data_loader = data_loader
        self.dataset_ids = dataset_ids
        self.output_txt = output_txt
        self.compile_txt = compile_txt

        self.dataset_info = self.dataset

        self.build_model()

        # # start with a pre-trained model
        if self.pretrained_model:
            self.load_pretrained_model()

        rand_seed = 64678  
        if rand_seed is not None:
            np.random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed(rand_seed)


    def build_model(self):
        """
        Instantiates the model, loss criterion, and optimizer
        """

        # instantiate model
        self.model_name = self.model
        self.model = get_model(self.model,
                               self.imagenet_pretrain,
                               self.model_save_path)
                            #    self.input_channels)

        # instantiate loss criterion
        # self.criterion = nn.MSELoss() 
        self.criterion = nn.L1Loss()

        # instantiate optimizer
        # if 'MARUNet' in self.model_name or 'ConNet' in self.model_name:
        #     self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        # else:
        #     self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
        #                             lr=self.lr)
        if self.model_name == 'CSRNet':
            self.optimizer = optim.SGD(self.model.parameters(), self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        # elif self.model_name == 'man':
        #     self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.model_name == 'CAN':
            if self.dataset_info == 'shanghaitech-b':
                self.optimizer = optim.Adam(self.model.parameters(), self.lr, weight_decay=self.weight_decay)

            elif self.dataset_info == 'shanghaitech-a' or self.dataset_info == 'ucf-cc-50' or self.dataset_info == 'ucf-qnrf':
                self.optimizer = optim.SGD(self.model.parameters(), self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.model_name == 'ConNet':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),lr=self.lr)
            # self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        elif self.model_name == 'MAN':
            self.optimizer = optim.Adam(self.model.parameters(), self.lr, weight_decay=self.weight_decay)
        
        # print network
        self.print_network(self.model, self.model_name)
        print(self.optimizer)

        # use gpu if enabled
        if torch.cuda.is_available() and self.use_gpu:
            self.model.cuda()
            self.criterion.cuda()

    def print_network(self, model, name):
        """
        Prints the structure of the network and the total number of parameters

        Arguments:
            model {Object} -- the model to be used
            name {str} -- name of the model
        """

        num_params = 0
        for name, param in model.named_parameters():
            if 'transform' in name:
                continue
            num_params += param.data.numel()
        write_print(self.output_txt, name)
        write_print(self.output_txt, str(model))
        write_print(self.output_txt,
                    'The number of parameters: {}'.format(num_params))

    # TODO
    def load_pretrained_model(self):
        """
        loads a pre-trained model from a .pth or .pth.tar file
        """

        # if pretrained model is a .pth file, load weights directly
        if ".pth.tar" not in self.pretrained_model:
            self.pretrained_model = self.pretrained_model.replace('.pth', '')
            self.model.load_state_dict(torch.load(os.path.join(
                self.model_save_path, '{}.pth'.format(self.pretrained_model))), strict=False)
        
        # if pretrained model is a .pth.tar file, load weights stored in 'state_dict' and 'optimizer' keys
        else:
            weights = torch.load(os.path.join(
                self.model_save_path, '{}'.format(self.pretrained_model)))
            self.model.load_state_dict(weights['state_dict'], strict=False)

            if self.mode == 'train' and 'optimizer' in weights.keys():
                self.optimizer.load_state_dict(weights['optimizer'])

        write_print(self.output_txt,
                    'loaded trained model {}'.format(self.pretrained_model))

    def print_loss_log(self, start_time, iters_per_epoch, e, i, loss):
        """
        Prints the loss and elapsed time for each epoch
        
        Arguments:
            start_time {float} -- time (milliseconds) at which training of an epoch began
            iters_per_epoch {int} -- number of iterations in an epoch
            e {int} -- current epoch
            i {int} -- current iteraion
            loss {float} -- loss value
        """

        total_iter = self.num_epochs * iters_per_epoch
        cur_iter = e * iters_per_epoch + i

        elapsed = time.time() - start_time
        total_time = (total_iter - cur_iter) * elapsed / (cur_iter + 1)
        epoch_time = (iters_per_epoch - i) * elapsed / (cur_iter + 1)

        epoch_time = str(datetime.timedelta(seconds=epoch_time))
        total_time = str(datetime.timedelta(seconds=total_time))
        elapsed = str(datetime.timedelta(seconds=elapsed))

        log = "Elapsed {}/{} -- {}, Epoch [{}/{}], Iter [{}/{}], " \
              "loss: {:.15f}".format(elapsed,
                                    epoch_time,
                                    total_time,
                                    e + 1,
                                    self.num_epochs,
                                    i + 1,
                                    iters_per_epoch,
                                    loss)

        write_print(self.output_txt, log)

    def save_model(self, e):
        """
        Saves the model and optimizer weights per e epoch

        Arguments:
            e {int} -- current epoch
        """
        path = os.path.join(
            self.model_save_path,
            '{}/{}.pth'.format(self.version, e+1)
        )

        # torch.save(self.model.state_dict, path)
        torch.save({'state_dict': copy.deepcopy(self.model.state_dict()),
            'optimizer': copy.deepcopy(self.optimizer.state_dict())
            }, path)

    def train_collate(batch):
        transposed_batch = list(zip(*batch))
        images = torch.stack(transposed_batch[0], 0)
        points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
        targets = transposed_batch[2]
        st_sizes = torch.FloatTensor(transposed_batch[3])
        return images, points, targets, st_sizes

    def model_step(self, images, targets, epoch):
        """
        A step for each iteration
        
        Arguments:
            images {torch.Tensor} -- input images
            targets {torch.Tensor} -- groundtruth density maps
            epoch {int} -- current epoch
        """

        # set model in training mode
        self.model.train()

        # empty the gradients of the model through the optimizer
        self.optimizer.zero_grad()

        # forward pass
        images = images.float()
        output = self.model(images)

        # if model is ConNet, prepare groundtruth attention maps
        if 'MARUNet' in self.model_name or 'ConNet' in self.model_name:
            if 'ConNet' in self.model_name:
                _, output = output
            output, d0, d1, d2, d3, d4, amp41, amp31, amp21, amp11, amp01 = output
            amp_gt = [get_amp_gt_by_value(l) for l in targets]
            amp_gt = torch.stack(amp_gt).cuda()
            
        # compute loss
        # if model is ConNet 
        if 'ConNet' in self.model_name:
            loss = 0
            target = 50 * targets[0].float().unsqueeze(1).cuda()
            
            # get losses of density maps
            outputs = [output, d0, d1, d2, d3, d4]
            for out in outputs:
                loss += cal_avg_ms_ssim(out, target, 3)

            # get losses of attention maps
            amp_outputs = [amp41, amp31, amp21, amp11, amp01]
            for amp in amp_outputs:
                amp_gt_us = amp_gt[0].unsqueeze(0)
                amp = amp.cuda()
                if amp_gt_us.shape[2:]!=amp.shape[2:]:
                    amp_gt_us = F.interpolate(amp_gt_us, amp.shape[2:], mode='bilinear')
                cross_entropy = (amp_gt_us * torch.log(amp+1e-10) + (1 - amp_gt_us) * torch.log(1 - amp+1e-10)) * -1
                cross_entropy_loss = torch.mean(cross_entropy)
                loss = loss + cross_entropy_loss * 0.1

            # compute gradients using back propagation
            loss.backward()
        else:
            loss = self.criterion(output.squeeze(), targets.squeeze())
        
            # compute gradients using back propagation
            loss.backward()

        # update parameters
        self.optimizer.step()

        # return loss
        return loss

    def man_model_step(self, images, targets, epoch, points, st_sizes):
        """
        A step for each iteration
        
        Arguments:
            images {torch.Tensor} -- input images
            targets {torch.Tensor} -- groundtruth density maps
            epoch {int} -- current epoch
        """

        # set model in training mode
        self.model.train()

        # empty the gradients of the model through the optimizer
        self.optimizer.zero_grad()

        # forward pass
        images = images.float()
        output = self.model(images)

        gd_count = np.array([len(p) for p in points], dtype=np.float32)

        with torch.set_grad_enabled(True):
            output, features = self.model(images)
            # compute loss using MSE loss function\
            
            # outputs, features = self.model(images)
            prob_list = self.post_prob(points, st_sizes)
            loss = self.criterion(prob_list, targets, output)
            loss_c = 0
            for feature in features:
                mean_feature = torch.mean(feature, dim=0)
                mean_sum = torch.sum(mean_feature**2)**0.5
                cosine = 1 - torch.sum(feature*mean_feature, dim=1) / (mean_sum * torch.sum(feature**2, dim=1)**0.5 + 1e-5)
                loss_c += torch.sum(cosine)
            loss += loss_c

            self.optimizer.zero_grad()
            loss.backward()
        
            # update parameters
            self.optimizer.step()

            N = images.size(0)
            pre_count = torch.sum(output.view(N, -1), dim=1).detach().cpu().numpy()
            res = pre_count - gd_count
            self.epoch_loss.update(loss.item(), N)
            self.epoch_mse.update(np.mean(res * res), N)
            self.epoch_mae.update(np.mean(abs(res)), N)

        # return loss
        return loss

    def train(self):
        """
        Performs training process
        """
        self.losses = []
        iters_per_epoch = len(self.data_loader)
        sched = 0

        # TODO
        # # start with a trained model if exists
        # if self.pretrained_model:
        #     try:
        #         start = int(self.pretrained_model.split('/')[-1].replace('.pth.tar', ''))
        #     except:
        #         start = 0

        #     for x in self.learning_sched:
        #         if start >= x:
        #             sched +=1
        #             self.lr /= 10
        #         else:
        #             break

        #     print("LEARNING RATE: ", self.lr, sched, " | EPOCH:", start)
        # else:
        #     start = 0

        start = 0
        # start training
        start_time = time.time()
        if 'MAN' in self.model_name:
            self.model = vgg_c.vgg19_trans()
            self.model.to(self.device)

            self.post_prob = Post_Prob(self.sigma,
                                self.crop_size,
                                self.downsample_ratio,
                                self.background_ratio,
                                self.use_background,
                                self.device)

            self.criterion = Bay_Loss(self.use_background, self.device)
            # self.save_list = Save_Handle(max_num=args.max_model_num)
            self.best_mae = np.inf
            self.best_mse = np.inf
            # self.save_all = args.save_all
            self.best_count = 0
            
            for e in range(start, self.num_epochs):
                logging.info('-'*5 + 'Epoch {}/{}'.format(e, self.num_epochs - 1) + '-'*5)
                self.epoch = e
        
                self.epoch_loss = AverageMeter()
                self.epoch_mae = AverageMeter()
                self.epoch_mse = AverageMeter()
                self.epoch_start = time.time()
                self.model.train()  # Set model to training mode
                    
                for i, (images, points, targets, st_sizes) in enumerate(tqdm(self.data_loader)):
                    # # prepare input images
                    # images = to_var(images, self.use_gpu)

                    # # prepare groundtruth targets
                    # targets = [to_var(torch.Tensor(target), self.use_gpu) for target in targets]
                    # targets = torch.stack(targets)

                    # train model and get loss
                    images = images.to(self.device)
                    st_sizes = st_sizes.to(self.device)
                    points = [p.to(self.device) for p in points]
                    targets = [t.to(self.device) for t in targets]

                    loss = self.man_model_step(images, targets, e, points, st_sizes)

                # # print out loss log
                # if (e + 1) % self.loss_log_step == 0:
                #     self.print_loss_log(start_time, iters_per_epoch, e, i, loss)
                #     self.losses.append((e, loss))

                

                # # update learning rate based on learning schedule
                # num_sched = len(self.learning_sched)
                # if num_sched != 0 and sched < num_sched:
                #     if (e + 1) in self.learning_sched:
                #         self.lr /= 10
                #         print('Learning rate reduced to', self.lr)
                #         sched += 1

                logging.getLogger().setLevel(logging.INFO)
                logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, self.epoch_loss.get_avg(), np.sqrt(self.epoch_mse.get_avg()), self.epoch_mae.get_avg(),
                             time.time()-self.epoch_start))
                model_state_dic = self.model.state_dict()
                save_path = os.path.join(self.model_save_path, '{}_ckpt.tar'.format(self.epoch))

                # save model
                if (e + 1) % self.model_save_step == 0:
                    self.save_model(e)
                    # torch.save({
                    #     'epoch': self.epoch,
                    #     'optimizer_state_dict': self.optimizer.state_dict(),
                    #     'model_state_dict': model_state_dic
                    # }, save_path)
                    # self.save_list.append(save_path)  # control the number of saved models
               
        else:
            for e in range(start, self.num_epochs):
                for i, (images, targets) in enumerate(tqdm(self.data_loader)):
                    # prepare input images
                    images = to_var(images, self.use_gpu)

                    # prepare groundtruth targets
                    targets = [to_var(torch.Tensor(target), self.use_gpu) for target in targets]
                    targets = torch.stack(targets)

                    loss = self.model_step(images, targets, e)

                # print out loss log
                if (e + 1) % self.loss_log_step == 0:
                    self.print_loss_log(start_time, iters_per_epoch, e, i, loss)
                    self.losses.append((e, loss))

                # save model
                if (e + 1) % self.model_save_step == 0:
                    self.save_model(e)

                # update learning rate based on learning schedule
                num_sched = len(self.learning_sched)
                if num_sched != 0 and sched < num_sched:
                    if (e + 1) in self.learning_sched:
                        self.lr /= 10
                        print('Learning rate reduced to', self.lr)
                        sched += 1

            # print losses
            write_print(self.output_txt, '\n--Losses--')
            for e, loss in self.losses:
                write_print(self.output_txt, str(e) + ' {:.10f}'.format(loss))

    def eval(self, data_loader):
        """
        Performs evaluation of a given model to get the MAE, MSE, FPS performance

        Arguments:
            data_loader {DataLoader} -- DataLoader of the dataset to be used
        """

        # set the model to eval mode
        self.model.eval()

        timer = Timer()
        elapsed = 0
        mae = 0
        mse = 0

        # predetermined save frequency of density maps
        save_freq = 100

        # begin evaluating on the dataset
        with torch.no_grad():
            for i, (images, targets) in enumerate(tqdm(data_loader)):
                # prepare the input images
                images = to_var(images, self.use_gpu)
                images = images.float()

                # prepare the groundtruth targets
                targets = [to_var(torch.Tensor(target), self.use_gpu) for target in targets]
                targets = torch.stack(targets)
                
                # generate output of model
                timer.tic()
                output = self.model(images)
                elapsed += timer.toc(average=False)

                # if model is ConNet, divide output by 50 as designed by original proponents
                if 'ConNet' in self.model_name:
                    output = output[0] / 50

                ids = self.dataset_ids[i*self.batch_size: i*self.batch_size + self.batch_size]
                model = self.pretrained_model.split('/')
                file_path = os.path.join(self.model_test_path, self.dataset_info + ' epoch ' + self.get_epoch_num())
                
                # generate copies of density maps as images 
                # if difference between predicted and actual counts are bigger than 1
                # if self.fail_cases:
                #     t = targets[0].cpu().detach().numpy()
                #     o = output[0].cpu().detach().numpy()

                #     gt_count = round(np.sum(t))
                #     et_count = round(np.sum(o))

                #     diff = abs(gt_count - et_count)

                #     if (diff > 0):
                #         save_plots(os.path.join(file_path, 'failure cases', str(diff)), output, targets, ids)
                
                # generate copies of density maps as images
                # if self.save_output_plots and i % save_freq == 0:
                #     save_plots(file_path, output, targets, ids)

                
                # update MAE and MSE (summation part of the formula)
                # mae += abs(output.sum() - targets.sum()).item()
                # mse += ((targets.sum() - output.sum())*(targets.sum() - output.sum())).item()
                
                # output = torch.stack(output, dim=0).sum(dim=0)
                mae += abs(output.sum() - targets.sum()).item()
                mse += ((targets.sum() - output.sum())*(targets.sum() - output.sum())).item()

        # compute for MAE, MSE and FPS
        mae = mae / len(data_loader)
        mse = np.sqrt(mse / len(data_loader))
        fps = len(data_loader) / elapsed

        return mae, mse, fps

    def man_eval(self, data_loader):
        """
        Performs evaluation of a given MAN model to get the MAE, MSE, FPS performance

        Arguments:
            data_loader {DataLoader} -- DataLoader of the dataset to be used
        """
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.save_all = self.save_all
        self.best_count = 0
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        # Iterate over data.
        for inputs, count, name in self.data_loader:
            inputs = inputs.to(self.device)
            # inputs are images with different sizes
            b, c, h, w = inputs.shape
            h, w = int(h), int(w)
            assert b == 1, 'the batch size should equal to 1 in validation mode'
            input_list = []
            if h >= 3584 or w >= 3584:
                h_stride = int(np.ceil(1.0 * h / 3584))
                w_stride = int(np.ceil(1.0 * w / 3584))
                h_step = h // h_stride
                w_step = w // w_stride
                for i in range(h_stride):
                    for j in range(w_stride):
                        h_start = i * h_step
                        if i != h_stride - 1:
                            h_end = (i + 1) * h_step
                        else:
                            h_end = h
                        w_start = j * w_step
                        if j != w_stride - 1:
                            w_end = (j + 1) * w_step
                        else:
                            w_end = w
                        input_list.append(inputs[:, :, h_start:h_end, w_start:w_end])
                with torch.set_grad_enabled(False):
                    pre_count = 0.0
                    for idx, input in enumerate(input_list):
                        output = self.model(input)[0]
                        pre_count += torch.sum(output)
                res = count[0].item() - pre_count.item()
                epoch_res.append(res)
            else:
                with torch.set_grad_enabled(False):
                    outputs = self.model(inputs)[0]
                    # save_results(inputs, outputs, self.vis_dir, '{}.jpg'.format(name[0]))
                    res = count[0].item() - torch.sum(outputs).item()
                    epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        logging.info('Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(mse, mae, time.time()-epoch_start))

        model_state_dic = self.model.state_dict()
        logging.info("best mse {:.2f} mae {:.2f}".format(self.best_mse, self.best_mae))
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            logging.info("save best mse {:.2f} mae {:.2f}".format(self.best_mse,
                                                                                 self.best_mae
                                                                                 ))
            if self.save_all:
                torch.save(model_state_dic, os.path.join(self.model_save_path, 'best_model_{}.pth'.format(self.best_count)))
                self.best_count += 1
            else:
                torch.save(model_state_dic, os.path.join(self.model_save_path, 'best_model.pth'))

        return mae, mse, 0.0

    def pred(self):

        # set the model to eval mode
        self.model.eval()
        data_loader = self.data_loader

        timer = Timer()
        elapsed = 0

        mae = 0
        mse = 0

        save_freq = 1

        if 'MAN' in self.model_name:
            # os.environ['CUDA_VISIBLE_DEVICES'] = self.device.strip()  # set vis gpu

            device = torch.device('cuda')
            model = vgg_c.vgg19_trans()
            model.to(device)
            model.eval()

            # model.load_state_dict(torch.load(self.pretrained_model, device))
            epoch_minus = []
            for i, (inputs, count, name, tar) in enumerate(tqdm(data_loader)):
                ids = self.dataset_ids[i*self.batch_size: i*self.batch_size + self.batch_size]
                inputs = inputs.to(device)
                b, c, h, w = inputs.shape
                h, w = int(h), int(w)
                assert b == 1, 'the batch size should equal to 1 in validation mode'
                input_list = []
                if h >= 3584 or w >= 3584:
                    h_stride = int(np.math.ceil(1.0 * h / 3584))
                    w_stride = int(np.math.ceil(1.0 * w / 3584))
                    h_step = h // h_stride
                    w_step = w // w_stride
                    for i in range(h_stride):
                        for j in range(w_stride):
                            h_start = i * h_step
                            if i != h_stride - 1:
                                h_end = (i + 1) * h_step
                            else:
                                h_end = h
                            w_start = j * w_step
                            if j != w_stride - 1:
                                w_end = (j + 1) * w_step
                            else:
                                w_end = w
                            input_list.append(inputs[:, :, h_start:h_end, w_start:w_end])
                    with torch.set_grad_enabled(False):
                        pre_count = 0.0
                        for idx, input in enumerate(input_list):
                            output = model(input)[0]
                            pre_count += torch.sum(output)
                    res = count[0].item() - pre_count.item()
                    epoch_minus.append(res)
                else:
                    with torch.set_grad_enabled(False):
                        outputs = model(inputs)[0]
                        res = count[0].item() - torch.sum(outputs).item()
                        epoch_minus.append(res)

                file_path = os.path.join(self.model_test_path, '{} {} epoch {}'.format(self.model_name, self.dataset_info, self.get_epoch_num()))
                file_path = "./" + file_path.replace('.pth', '')

                if self.save_output_plots and i % save_freq == 0:
                    # prepare the groundtruth targets
                    targets = [to_var(torch.Tensor(x), self.use_gpu) for x in tar]
                    targets = torch.stack(targets)
                    save_plots(file_path, outputs, targets, ids, save_label=False)
                    # save_plots(file_path, output, [], ids, pred=True)

            epoch_minus = np.array(epoch_minus)
            print(epoch_minus)
            mse = np.sqrt(np.mean(np.square(epoch_minus)))
            mae = np.mean(np.abs(epoch_minus))
            log_str = 'mae {}, mse {}'.format(mae, mse)
            print(log_str)

            
        else:
            with torch.no_grad():
                for i, (images, targets) in enumerate(tqdm(data_loader)):
                    images = to_var(images, self.use_gpu)
                    images = images.float()

                    timer.tic()
                    output = self.model(images)
                    elapsed += timer.toc(average=False)

                    ids = self.dataset_ids[i*self.batch_size: i*self.batch_size + self.batch_size]

                    if 'ConNet' in self.model_name:
                        output = output[0] / 50

                    model = self.pretrained_model.split('/')
                    file_path = os.path.join(self.model_test_path, '{} {} epoch {}'.format(self.model_name, self.dataset_info, self.get_epoch_num()))
                    file_path = "./" + file_path.replace('.pth', '')

                    if self.save_output_plots and i % save_freq == 0:
                        # prepare the groundtruth targets
                        targets = [to_var(torch.Tensor(target), self.use_gpu) for target in targets]
                        targets = torch.stack(targets)
                        save_plots(file_path, output, targets, ids, save_label=True)
                        # save_plots(file_path, output, [], ids, pred=True)

    def test(self):
        """
        Evaluates the performance of the model using the test dataset
        """

        # evaluate the model
        if 'MAN' in self.model_name:
            out = self.man_eval(self.data_loader)
        else:
            out = self.eval(self.data_loader)

        # log the performance
        log = ('mae: {:.6f}, mse: {:.6f}, '
               'fps: {:.4f}')
        log = log.format(out[0], out[1], out[2])
        write_print(self.output_txt, log)

        epoch_num = self.get_epoch_num()
        write_to_file(self.compile_txt, 'epoch {} | {}'.format(epoch_num, log))

        try:
            if (int(epoch_num) % 5 == 0):
                write_to_file(self.compile_txt, '')
        except:
            pass

    def get_epoch_num(self):
        '''
        Gets the epoch number given the format of the pretrained model's file name
        '''

        if 'SKT experiments' in self.model_test_path:
            epoch_num = self.output_txt.split('/')[-1]
            epoch_num = epoch_num[epoch_num.rfind('epoch_')+6:epoch_num.rfind('_mae')]
        else:
            epoch_num = self.output_txt[self.output_txt.rfind('_')+1:-4].replace('.pth.tar', '')

        return epoch_num
