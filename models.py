import torch.nn as nn
import torch.nn.functional as F 
import torch
import torch as th


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.zero_pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(kernel_size=3, in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.bnorm = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.zero_pad(x)
        x = self.conv(x)
        x = self.bnorm(x)
        x = F.relu(x)
        return x



class Yolo(nn.Module):
    def __init__(
        self, 
        conv_list, 
        nc_last_conv, 
        device,
        n_classes=91,
        n_anchors=1,
        final_grid_size_i=9,
        final_grid_size_j=9,
        last_kernel_dim=30
        ):

        super().__init__()
        self.conv_list = conv_list
        self.last_conv = nn.Conv2d(kernel_size=1, in_channels=conv_list[-1][0], out_channels=nc_last_conv, stride=1)
        conv_layers = []
        for (in_channels, _), (out_channels, stride) in zip(conv_list[:-1], conv_list[1:]):
            conv_layers.append(ConvLayer(in_channels, out_channels, stride).to(device))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.default_anchor_size = 0.3
        self.n_classes = n_classes
        self.n_anchors = n_anchors

    def forward(self, x, separate=True):
        for layer, (inc, _), (outc, stride) in zip(self.conv_layers, self.conv_list[:-1], self.conv_list[1:]):
            skip = x
            x = layer(x)
            # print(x.shape)
            if inc==outc and stride==1:
                x += skip
        
        # print(x.shape)
        x = self.last_conv(x)
        # print(x.shape)
        tmp = self.n_classes+1
        tmp2 = tmp  + self.n_anchors*2
        tmp3 = tmp2 + self.n_anchors*2

        class_preds = x[:, :self.n_classes,   :, :]
        objectness  = x[:, self.n_classes,    :, :].unsqueeze(1)
        offsets     = x[:, tmp :tmp2, :, :]
        boxes       = x[:, tmp2:tmp3, :, :]

        objectness  = th.sigmoid(objectness)
        boxes       = th.exp(boxes) * self.default_anchor_size
        offsets     = th.sigmoid(offsets) 
        class_preds = th.sigmoid(class_preds)
        # print("-"*10)
        if separate:
            return class_preds, objectness, offsets, boxes
        return torch.cat((class_preds, objectness, offsets, boxes), dim=1)



class Detector(nn.Module):
    def __init__(self, conv_list, device, backbone_strings=('pytorch/vision:v0.6.0', 'mobilenet_v2'), backbone_train=False):
        super().__init__()
        self.backbone = torch.hub.load(backbone_strings[0], backbone_strings[1], pretrained=True).features.to(device)
        if not backbone_train:
            self.backbone.eval()
            for p in self.backbone:
               p.requires_grad = False
        self.head = Yolo(conv_list, LAST_KERNEL_DIM, device=device).to(device)
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)