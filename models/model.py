import torch
import os
import torch.nn as nn

from models.CSRNet.CSRNet import CSRNet
from models.CAN.CAN import CANNet
from models.ConNet.ConNet import ConNet
from models.MAN.MAN import MAN
# from models.CSRNet.model_student_vgg import CSRNet as CSRNetSKT

# from models.MCNN.network import weights_normal_init
# from models.MCNN.crowd_count import CrowdCounter

# from models.MARUNet.marunet import MARNet
# from models.MARUNet.student_marunet import MARNet as MARNetSKT

# from models.MUSCO.musco_marunet_mall import MARNet as MARNetMUSCO_mall
# from models.MUSCO.musco_marunet_micc import MARNet as MARNetMUSCO_micc
# from models.MUSCO.musco_csrnet_mall import CSRNet as CSRNetMUSCO_mall
# from models.MUSCO.musco_csrnet_micc import CSRNet as CSRNetMUSCO_micc

# from models.ConNet.connet_mall import ConNet as ConNet_mall
# from models.ConNet.connet_micc import ConNet as ConNet_micc


def init_weights(model, classifier_only=False):
    # print(model)

    for module in model.modules():
        if isinstance(module, nn.Conv2d) and not classifier_only:
            nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                    nonlinearity='relu')

        elif isinstance(module, nn.BatchNorm2d) and not classifier_only:
            nn.init.constant_(module.weight, val=1)
            nn.init.constant_(module.bias, val=0)

        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.01)
            nn.init.constant_(module.bias, val=0)

# used for MAN
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# def load_pretrained_model(model, model_save_path, pretrained_model):
#     """
#     loads a pre-trained model from a .pth file
#     """
#     model.load_state_dict(torch.load(os.path.join(
#         model_save_path, '{}.pth'.format(pretrained_model))))


def get_model(model_config,
              imagenet_pretrain,
              model_save_path,
            #   input_channels,
              mode="train"):

    model = None

    if model_config == "CSRNet":
        model = CSRNet()

    elif model_config == "CAN":
        model = CANNet()

    elif model_config == "ConNet":
        model = ConNet(transform=mode.lower() == "train")
    
    elif model_config == "MAN":
        cfg = {'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}

        model = MAN(make_layers(cfg['E']))

    # if model_config == "ConNet_mall" or model_config == "ConNet_04":
    #     model = ConNet_mall(transform=mode.lower() == "train")

    # elif model_config == "ConNet_micc" or model_config == "ConNet_08":
    #     model = ConNet_micc(transform=mode.lower() == "train")

    # elif model_config == "CSRNet":
    #     model = CSRNet()
    # elif model_config == "CSRNetSKT":
    #     model = CSRNetSKT(ratio=4, transform=mode.lower() == "train")

    # elif model_config == "MCNN":
    #     model = CrowdCounter()
    #     weights_normal_init(model, dev=0.01)

    # elif model_config == "MARUNet":
    #     # torch.backends.cudnn.enabled = False
    #     model = MARNet(objective='dmp+amp')
    # elif model_config == "MARUNetSKT":
    #     model  = MARNetSKT(ratio=4, bn=True, transform=mode.lower() == "train")

    # elif "MUSCO" in model_config.upper():
    #     if model_config == "MARUNetMUSCO_mall":
    #         model = MARNetMUSCO_mall(objective='dmp+amp')
    #     elif model_config == "MARUNetMUSCO_micc":
    #         model = MARNetMUSCO_micc(objective='dmp+amp')
    #     elif model_config == "CSRNetMUSCO_mall":
    #         model = CSRNetMUSCO_mall()
    #     elif model_config == "CSRNetMUSCO_micc":
    #         model = CSRNetMUSCO_micc()

    #     if "x_iter" in model_config:
    #         import importlib
    #         if "MARUNet" in model_config:
    #             module = importlib.import_module('models.MUSCO.iter_exp.{}'.format(model_config.lower()))
    #             model = module.MARNet(objective='dmp+amp')
    #         elif "CSRNet" in model_config:
    #             module = importlib.import_module('models.MUSCO.iter_exp.{}'.format(model_config.lower()))
    #             model = module.CSRNet()


    if imagenet_pretrain is not True:
        init_weights(model)
    else:
        init_weights(model, classifier_only=True)

    return model
