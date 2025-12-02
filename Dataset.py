import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import random
from torchvision import transforms
from torchvision.transforms.functional import crop
import matplotlib.pyplot as plt
# 数据加载
class CustomDataset(Dataset):
    def __init__(self, station_path, transform=None):
        self.station_path = station_path
        self.image_names = os.listdir(station_path)
        self.num_images = len(self.image_names)
        self.transform = transform

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.station_path, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            input_data = self.transform(image)

        label = int(image_name.split('_')[0])
        image_tensor = torch.tensor(np.array(image).transpose((2, 0, 1)))
        return image_tensor, input_data, label


class RandomCropAndTransform:
    def __init__(self):
        self.crop_size = 0

    def __call__(self, image):
        width, height = image.size
        self.crop_size = int(min(width, height) * 0.95)

        # 随机选择两个区域
        x1 = random.randint(0, width - self.crop_size)
        y1 = random.randint(0, height - self.crop_size)
        x2 = random.randint(0, width - self.crop_size)
        y2 = random.randint(0, height - self.crop_size)

        # 随机水平反转
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)


        # 裁剪并转换为tensor
        cropped_image1 = crop(image, y1, x1, self.crop_size, self.crop_size)
        cropped_image2 = crop(image, y2, x2, self.crop_size, self.crop_size)

        return cropped_image1, cropped_image2

    def enhance_img_imshow(self, image_tensor1, image_tensor2):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image_tensor1.permute(1, 2, 0).numpy())
        plt.subplot(1, 2, 2)
        plt.imshow(image_tensor2.permute(1, 2, 0).numpy())
        plt.show()


class PairedDataset(Dataset):
    def __init__(self, station_train, trainPath, transform=None):
        """
        初始化成对数据集。

        参数:
        - station_train (list): 训练集中场景的列表，包含20个场景名。
        - trainPath (str): 存储所有场景的根路径。
        - transform (callable, optional): 应用于图片的变换。
        """
        self.station_train = station_train
        self.trainPath = trainPath
        self.transform = transform
        self.paired_data = []  # 存储所有有效的图片对

        for station in self.station_train:
            station_path = os.path.join(self.trainPath, station)
            image_names = sorted(os.listdir(station_path))  # 确保顺序一致
            labels = []
            image_paths = []
            for image_name in image_names:
                try:
                    label = int(image_name.split('_')[0])
                except (IndexError, ValueError) as e:
                    raise ValueError(f"无法从文件名 {image_name} 中提取标签。") from e
                labels.append(label)
                image_paths.append(os.path.join(station_path, image_name))

            # 找到所有符合条件的图片对
            valid_pairs = []
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if 0 < abs(labels[i] - labels[j]) < 3:
                        valid_pairs.append((i, j))

            if not valid_pairs:
                raise ValueError(f"场景 {station} 中没有满足标签差值大于10的图片对。")

            # 随机选择一个有效的图片对
            selected_pair = random.choice(valid_pairs)
            i, j = selected_pair

            # 记录图片对的信息
            self.paired_data.append({
                'station': station,
                'image1_path': image_paths[i],
                'label1': labels[i],
                'image2_path': image_paths[j],
                'label2': labels[j],
                'label_diff': labels[i] - labels[j]
            })

    def __len__(self):
        return len(self.paired_data)

    def __getitem__(self, idx):
        """
        获取指定索引的图片对及其标签差值。

        参数:
        - idx (int): 数据索引。

        返回:
        - images (tuple): (img1_tensor, img2_tensor), each [3, 256, 256]
        - label_diff (float): label1 - label2
        """
        pair = self.paired_data[idx]
        img1 = Image.open(pair['image1_path']).convert('RGB')
        img2 = Image.open(pair['image2_path']).convert('RGB')

        if self.transform:
            input1 = self.transform(img1)
            input2 = self.transform(img2)
        else:
            input1 = torch.from_numpy(np.array(img1).transpose((2, 0, 1))).float() / 255.0
            input2 = torch.from_numpy(np.array(img2).transpose((2, 0, 1))).float() / 255.0

        label_diff = pair['label_diff']

        return (input1, input2), label_diff


