import torch.nn.functional as F

from models.resnet18_encoder import *
from models.resnet20_cifar import *


class MYNET(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        # self.num_features = 512
        if self.args.dataset in ['cifar100']:
            self.feature_extractor = resnet20(num_classes=100)
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet']:
            self.feature_extractor = resnet18(False, args, num_classes=100)
            self.num_features = 512
        if self.args.dataset == 'cub200':
            self.feature_extractor = resnet18(True,
                                              args,
                                              num_classes=200)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512

    def forward(self, *args):
        data = args[0]
        data = self.feature_extractor(data)
        logits = F.linear(F.normalize(data, p=2, dim=-1), F.normalize(self.feature_extractor.fc.weight, p=2, dim=-1))
        logits = self.args.temperature * logits
        # logits = self.feature_extractor.fc(data)
        return logits
