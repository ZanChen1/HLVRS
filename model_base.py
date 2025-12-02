import torch
import torch.nn as nn
import torch.nn.parallel
from get_feature.Unet import UNet as UNet
from config import Common, Model_choose, Img_message, Index_error_record
from Depth_model.Revisiting_Single_Depth_Estimation.models import modules, nnet, resnet, densenet, senet

class VisibilityCl_base(nn.Module):
    def __init__(self):
        super(VisibilityCl_base, self).__init__()
        self.feature_model = self.get_feature()
        self.get_depth_map = self.get_depth()
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16384, 1000)
        self.grnn = GRNN(1000)


        # 深度特征卷积
        self.depth_layers = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        )

    def forward(self, x):
        feature_map = self.feature_model(x).to(Common.device)
        Index_error_record.feature_map = feature_map

        # 深度特征处理
        depth_map, x_decoder, x_mff = self.get_depth_map(x)
        x_decoder.to(Common.device)
        x_mff.to(Common.device)

        depth_map = self.depth_layers(torch.cat((x_decoder, x_mff), 1))

        map_concat = torch.cat((depth_map, feature_map), dim=1)
        GRNN_input = self.conv1(map_concat)
        GRNN_input = torch.flatten(GRNN_input, 1)
        GRNN_input = self.relu(GRNN_input)
        GRNN_input = self.fc(GRNN_input)
        score = self.grnn(GRNN_input)
        score = self.normalize_tensor(score)
        return score

    def get_feature(self):
        feature_model = UNet()
        feature_model.load_state_dict(
            torch.load('./model/val/unet_carvana_scale1.0_epoch2.pth',
                   map_location='cuda:0'), strict=False)
        feature_model.to(Common.device)
        return feature_model

    def get_depth(self):
        original_model = resnet.resnet50(pretrained=True)
        Encoder = modules.E_resnet(original_model)
        depth_model = nnet.model(Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048])
            # 设置 depth_model 的参数不可训练
        for param in depth_model.parameters():
            param.requires_grad = False
        depth_model.load_state_dict(
            torch.load('./Depth_model/Revisiting_Single_Depth_Estimation/pretrained_model/model_senet',
                           map_location='cuda:0'), strict=False)
        depth_model.eval()
        depth_model = depth_model.to(Common.device)

        return depth_model

    def normalize_tensor(self, input_tensor, method='standard'):
        """
        Normalize a 2D tensor along the columns.

        Parameters:
        - input_tensor: 2D PyTorch tensor
        - method: str, normalization method ('standard' or 'min_max')

        Returns:
        - normalized_tensor: 2D PyTorch tensor
        """
        if method == 'standard':
            # Standard normalization (z-score normalization)
            mean_value = torch.mean(input_tensor, dim=0)
            std_value = torch.std(input_tensor, dim=0)
            normalized_tensor = (input_tensor - mean_value) / std_value
        elif method == 'min_max':
            # Min-Max normalization
            min_value = torch.min(input_tensor, dim=0).values
            max_value = torch.max(input_tensor, dim=0).values
            normalized_tensor = (input_tensor - min_value) / (max_value - min_value)
        else:
            raise ValueError("Invalid normalization method. Use 'standard' or 'min_max'.")

        return normalized_tensor

class GRNN(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(GRNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)  # 输出层只有一个节点

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def VisNet():
    model = VisibilityCl_base()
    return model
