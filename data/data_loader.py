import torch
# from data.mall_dataset import MallDataset
from data.shanghaitech_a import ShanghaiTechA
# from data.fdst import FDST
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from data.crowd import Crowd
from models.MAN import vgg_c
# from data.augmentations import Augmentations, BaseTransform
import numpy as np
import logging
import os

def collate(batch):
    """Collate function used by the DataLoader"""

    images = []
    targets = []
    for sample in batch:
        images.append(sample[0])
        # converted to float 32 to resolve type error
        targets.append(torch.FloatTensor(np.float32(sample[1])))
    return torch.stack(images, 0), targets

def man_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes

def get_loader(config):
    """Returns the data loader and dataset image ids

    Arguments:
        config {dict} -- contains necessary arguments and its values for the 
            instantiation of the DataLoader object

    Returns:
        DataLoader -- DataLoader object of the specified dataset to be used
        list -- list of image IDs in the dataset 
    """
    
    dataset = None
    loader = None
    targets_resize = 1
    image_transform = None

    # targets_resize refers to how much the dimensions of the
    # target density map must be downscaled to match the output
    # size of the model used
    if 'CSRNet' in config.model:
        targets_resize = 2 ** 3
    elif 'CAN' in config.model:
        targets_resize = 2 ** 3
    # elif config.model == 'MCNN':
    #     targets_resize = 2 ** 2

    # if config.augment_exp:
    #     image_transform = Augmentations(brightness=config.brightness_change,
    #         scale=config.resolution_scale)

    # get the Dataset object 
    if config.dataset == 'shanghaitech-a':
        if 'MAN' in config.model:
            dataset = ShanghaiTechA(data_path='../Datasets/ShanghaiTechAPreprocessed/',
                        mode=config.mode,                            
                        # image_transform=image_transform,
                        targets_resize=targets_resize)
        else:
            dataset = ShanghaiTechA(data_path=config.shanghaitech_a_path,
                            mode=config.mode,                            
                            # image_transform=image_transform,
                            targets_resize=targets_resize)

    
    # get the data loader
    if dataset is not None:
        if 'MAN' in config.model:
            if torch.cuda.is_available():
                config.device = torch.device("cuda")
                config.device_count = torch.cuda.device_count()
                # for code conciseness, we release the single gpu version
                assert config.device_count == 1
                logging.info('using {} gpus'.format(config.device_count))
            else:
                raise Exception("gpu is not available")

            config.datasets = Crowd(os.path.join('C:/Users/lande/Desktop/THS-ST2/Datasets/ShanghaiTechAPreprocessed', config.mode),
                                config.crop_size,
                                config.downsample_ratio,
                                config.is_gray, config.mode)

            loader = DataLoader(dataset = config.datasets,
                                    collate_fn=(man_collate if config.mode == 'train' else default_collate),
                                    batch_size=(config.batch_size if config.mode == 'train' else 1),
                                    shuffle=(True if config.mode == 'train' else False),
                                    num_workers=config.num_workers*config.device_count,
                                    pin_memory=(True if config.mode == 'train' else False))
                    
        else:
            if config.mode == 'train':
                loader = DataLoader(dataset=dataset,
                                    batch_size=config.batch_size,
                                    shuffle=True,
                                    collate_fn=collate,
                                    num_workers=4,
                                    pin_memory=True)

            elif config.mode == 'val' or config.mode == 'test' or config.mode == 'pred':
                loader = DataLoader(dataset=dataset,
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    collate_fn=collate,
                                    num_workers=4,
                                    pin_memory=True)

    return loader, dataset.image_ids
