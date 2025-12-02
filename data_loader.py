# 自定义数据加载器
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import Common
from config import Train
import os
from PIL import Image
import torch.utils.data as Data
import numpy
import dill
# from openpyxl import load_workbook

# 定义数据处理transform
transform = transforms.Compose([
    transforms.Resize(Common.imageSize),
    transforms.ToTensor()
])


# 训练集
def loadDataFromDir():
    '''
    从文件夹中获取数据
    '''
    inputs = []
    labels = []
    n = 0
    station_train = os.listdir(Common.trainPath)
    num_station = len(station_train)
    for station in station_train:
        n += 1
        imageNames = os.listdir(Common.trainPath + station)
        print('!!!!!!!!!!!!!!!!!!!!!!\n')
        num_image = len(imageNames)
        k = 0
        num = num_image * (num_image - 1) / 2
        # 外层遍历
        for i in range(num_image - 1):
            image1Nmae = imageNames[i]
            tag1 = int(image1Nmae.split('_')[0])
            image1 = Image.open(Common.trainPath + station + "/" + image1Nmae).convert('RGB')
            # 内层遍历
            for j in range(i + 1, num_image):
                image2Nmae = imageNames[j]
                tag2 = int(image2Nmae.split('_')[0])
                image2 = Image.open(Common.trainPath + station + "/" + image2Nmae).convert('RGB')
                inp = torch.cat([transform(image1), transform(image2)], 0)
                inputs.append(inp)
                # 构造label
                label = [0] * 2  # 初始化label
                if tag1 > tag2:
                    label[0] = 1
                elif tag1 < tag2:
                    label[1] = 1
                label = torch.tensor(label, dtype=torch.float)  # 转为tensor张量
                # 6. 添加到目标值列表
                labels.append(label)
                # 7. 关闭资源
                image2.close()
                k += 1
                print("[" + str(n) + "/" + str(num_station) + "]" + "  正在加载“" + station + "”的第" + str(k) + "/" + str(
                    int(num)) + "条数据")
            image1.close()
    # 返回图片列表和目标值列表
    return inputs, labels


class WeatherDataSet(Dataset):
    '''
    自定义DataSet
    '''

    def __init__(self):
        '''
        初始化DataSet
        :param transform: 自定义转换器
        '''
        images, labels = loadDataFromDir()  # 在文件夹中加载图片
        print('-------------------------------------------------------------down------------------------------------\n')
        self.images = images
        self.labels = labels

    def __len__(self):
        '''
        返回数据总长度
        :return:
        '''
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


def splitData(dataset):
    '''
    分割数据集
    :param dataset:
    :return:
    '''
    # 求解一下数据的总量
    total_length = len(dataset)

    # 确认一下将80%的数据作为训练集, 剩下的20%的数据作为测试集
    train_length = int(total_length * 0.8)
    validation_length = total_length - train_length

    # 利用Data.random_split()直接切分数据集, 按照80%, 20%的比例进行切分
    train_dataset, validation_dataset = Data.random_split(dataset=dataset, lengths=[train_length, validation_length])
    return train_dataset, validation_dataset


def getDataLoader(loadFromLocal=True):
    trainLoaderPath = Common.dataloaderPath + "trainLoader.pkl"
    valLoaderPath = Common.dataloaderPath + "valLoader.pkl"

    if loadFromLocal:
        # 本地导入数据器加载器
        print("从本地导入数据器加载器")
        with open(trainLoaderPath, "rb") as f:
            trainLoader = dill.load(f)
        with open(valLoaderPath, "rb") as f:
            valLoader = dill.load(f)
    else:
        # 生成本地数据器加载器
        # 1. 分割数据集
        train_dataset, validation_dataset = splitData(WeatherDataSet())
        # 2. 训练数据集加载器
        trainLoader = DataLoader(train_dataset, batch_size=Train.batch_size, shuffle=True,
                                 num_workers=Train.num_workers)
        with open(trainLoaderPath, "wb") as f:
            dill.dump(trainLoader, f)
        # 3. 验证集数据加载器
        valLoader = DataLoader(validation_dataset, batch_size=Train.batch_size, shuffle=False,
                               num_workers=Train.num_workers)
        with open(valLoaderPath, "wb") as f:
            dill.dump(valLoader, f)
    return trainLoader, valLoader
