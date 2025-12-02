import time
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np



class Model_choose:
    """
    模型选择
    """
    Depth_choose = True  # 深度网络
    Seg_choose = True  # 分割网络


# 项目配置文件
class Common:
    '''
    通用配置
    '''
    # trainPath = "./data/train/"  # 训练集路径
    trainPath = "./data/train_new/"  # 训练集路径
    train_MRFID = "./data/train_MRFID/"
    test_MRFID = "./data/test_MRFID/"
    testPath = "./data/12-20(new)/"  # 测试集路径
    # testPath = "./data/test/"  # 测试集路径
    testone = "./data/one/"
    refPath = "./refs/"  # 参考图像路径
    resultPath = "./result/" # 可视化保存路径
    dataloaderPath = "./dataloader/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设备配置
    imageSize = (224, 224)  # 图片大小
    imageSize_depth_GRNN = (256, 256)
    labels = ["thick", "moderate", "light", "fogfree"]  # 标签名称/文件夹名称
    recPath = "./model/record/"
    train_testPath = "./data/train_test.xlsx"
    train_save_index = True  # 训练控制保存
    test_save_index = False  # 测试控制保存
    batch_number = 40
    enhance_index = True
    sigma = 2
    record_filename = ""


class Img_message:
    station = "之江大桥"
    label = 0
    save_choose_station = "洋溪23016雅鼎路边"
    save_choose_label = 10


class Train:
    '''
    训练相关配置
    '''
    mix_index = False
    batch_size = 32
    num_workers = 8  # 对于Windows用户，这里应设置为0，否则会出现多线程错误
    lr = 0.00012
    epochs = 100
    logDir = "./log/" + time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())  # 日志存放位置
    modelDir = "./model/val/"  # 验证模型存放位置
    model_pre_trained = "./model/pre_trained/"
    model_train_logpath = "./model/training_records/"
    minLoss = 10
    mixIndex = 7

class Test_index:
    accuracy0 = 0
    accuracy1 = 1
    accuracy2 = 2


class Index_record:
    def __init__(self):
        '''
        初始化相关指标
        '''

        self.epoch = 0
        self.Loss = 0  # RankNetLoss进行反向传播
        self.LossMSE = 0
        self.Loss_name = ""
        self.Loss_test = 0  # 验证loss
        self.epochNDCG = 0
        self.epochNDCG_test = 0
        self.epochMRR = 0
        self.epochAcc = 0  # 每个epoch的准确率
        self.epochcorrectNum = 0  # 正确预测的数量
        self.R_squared = 0

class Index_error_record:
    # 记录内容
    save_index = True
    feature_map = []
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    x6 = []
    x7 = []
    x8 = []
    x9 = []
    x1_weight = []


