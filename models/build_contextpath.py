import torch
from torchvision import models
import torch.utils.model_zoo as model_zoo
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.us-west-1.wasabisys.com/encoding/models/resnet50s-a75c83cf.zip',
    'resnet101s': 'https://s3.us-west-1.wasabisys.com/encoding/models/resnet101s-03a0f310.zip',
    'resnet152': 'https://s3.us-west-1.wasabisys.com/encoding/models/resnet152s-36670e8b.zip',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


class resnet18(torch.nn.Module):
    def __init__(self, path_model=None):
        super().__init__()
        resnet18_model = models.resnet18()
        if path_model:
            resnet18_model.load_state_dict(torch.load(path_model, map_location="cpu"))
            print("load pretrained model , done!! ")
        else:
            resnet18_model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
            print("load pretrained model , done!! ")

        self.features = resnet18_model
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail


class resnet101(torch.nn.Module):
    def __init__(self, path_model=None):
        super().__init__()

        resnet101_model = models.resnet101()
        if path_model:
            resnet101_model.load_state_dict(torch.load(path_model, map_location="cpu"))
        else:
            resnet101_model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
            print("load pretrained model , done!! ")

        self.features = resnet101_model
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail


def build_contextpath(name, path_model=False):
    assert name in ["resnet18", "resnet101"], "{} is not support! please use resnet18 or resnet101".format(name)
    if name == "resnet18":
        model = resnet18(path_model=path_model)
    elif name == "resnet101":
        model = resnet101(path_model=path_model)
    else:
        # raise "backbone is not defined!"
        pass
    return model


if __name__ == '__main__':
    #
    model_18 = build_contextpath('resnet18')
    model_101 = build_contextpath('resnet101')
    x = torch.rand(1, 3, 256, 256)

    y_18 = model_18(x)
    y_101 = model_101(x)
    # print(y_18,y_101)