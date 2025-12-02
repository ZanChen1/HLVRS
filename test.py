import random
import time
import numpy as np
import torch
from config import Common, Train, Index_record, Test_index
from Dataset import CustomDataset,  RandomCropAndTransform
from model_base_depth_version2 import VisNet
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorchltr.loss import PairwiseHingeLoss
from pytorchltr.loss import PairwiseLogisticLoss
from pytorchltr.loss import LambdaARPLoss1
from pytorchltr.loss import LambdaARPLoss2
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr
from datetime import datetime, timezone, timedelta

# 设置时区为中国标准时间 (CST)
china_timezone = timezone(timedelta(hours=8))

# 获取当前时间并应用时区
china_time = datetime.now(china_timezone)

# 定义数据处理transform
transform = transforms.Compose([
    transforms.Resize(Common.imageSize_depth_GRNN),
    transforms.ToTensor()
])

def CalculateIndex(predictions, targets, station):
    N = torch.tensor([len(predictions)])
    predictions = predictions.transpose(1, 0)

    targets = targets.float().unsqueeze(0)

    sorted_predictions, predictions_rank = torch.sort(predictions)
    x, predictions_rank = torch.sort(predictions_rank)
    predictions_rank = predictions_rank + 1
    print(predictions_rank)

    kur = kurtosis_coefficient(predictions)
    IQR = calculate_iqr(predictions)
    CV = coefficient_of_variation(predictions)
    R_squared = r_squared(predictions_rank, targets)
    KRCC = calculate_krcc(predictions_rank, targets)
    SRCC = calculate_srcc(predictions_rank, targets)
    accuracy0, correct_count0 = compute_accuracy(targets, predictions_rank, Test_index.accuracy0)
    accuracy1, correct_count1 = compute_accuracy(targets, predictions_rank, Test_index.accuracy1)
    accuracy2, correct_count2 = compute_accuracy(targets, predictions_rank, Test_index.accuracy2)
    Lossfunction1 = PairwiseHingeLoss()
    Lossfunction2 = PairwiseLogisticLoss()
    Lossfunction3 = LambdaARPLoss1()
    Lossfunction4 = LambdaARPLoss2()
    HingeLoss = Lossfunction1(predictions, targets, N).mean().item()
    LogisticLoss = Lossfunction2(predictions, targets, N).mean().item()
    LambdaLoss1 = Lossfunction3(predictions, targets, N).mean().item()
    LambdaLoss2 = Lossfunction4(predictions, targets, N).mean().item()
    RMSE = rmse(predictions_rank, targets)
    plot_ranking_errors_line(predictions_rank, targets)

    Index_values = {
        'HingeLoss': HingeLoss,
        'LogisticLoss': LogisticLoss,
        'LambdaLoss1': LambdaLoss1,
        'LambdaLoss2': LambdaLoss2,
        'RMSE': RMSE.item(),
        'KRCC': KRCC,
        'SRCC': SRCC,
        'Accuracy0': accuracy0,
        'Correct_count0': correct_count0,
        'Accuracy1': accuracy1,
        'Correct_count1': correct_count1,
        'Accuracy2': accuracy2,
        'Correct_count2': correct_count2,
        'Kurtosis': kur.item(),
        'IQR': IQR,
        'R_squared': R_squared,
        'CV': CV
    }

    print(
        f"Station: {station}\n"
        f"Hinge Loss: {HingeLoss:.4f}\t"
        f"Logistic Loss: {LogisticLoss:.4f}\t"
        f"Lambda Loss 1: {LambdaLoss1:.4f}\t"
        f"Lambda Loss 2: {LambdaLoss2:.4f}\n"
        f"RMSE: {RMSE:.4f}\t"
        f"KRCC: {KRCC:.4f}\t"
        f"SRCC: {SRCC:.4f}\t"
        f"Accuracy: {accuracy0:.4f}({accuracy1:.4f}) ({accuracy2:.4f})\t"
        f"Correct_count: {correct_count1} ({correct_count2})\t"
        f"Kurtosis: {kur:.4f}\t"
        f"IQR: {IQR:.4f}\t"
        f"R_squared: {R_squared:.4f}\t"
        f"CV: {CV:.4f}\n"
    )

    return Index_values


def Test(model, Transform):
    station_test = os.listdir(Common.test_MRFID)


    all_hinge_loss = []
    all_logistic_loss = []
    all_lambda_loss1 = []
    all_lambda_loss2 = []
    all_rmse = []
    all_accuracy0 = []
    all_correct_count0 = []
    all_accuracy1 = []
    all_correct_count1 = []
    all_accuracy2 = []
    all_correct_count2 = []
    all_kurtosis = []
    all_iqr = []
    all_R_squared = []
    all_cv = []
    all_krcc = []
    all_srcc = []

    for station in station_test:
        # 创建数据集实例
        test_dataset = CustomDataset(
            station_path=os.path.join(Common.test_MRFID, station),
            transform=transform)
        savepath = Common.resultPath + china_time.strftime('%Y-%m-%d-%H-%M-%S') + "/"
        # 进行测试
        predictions, targets = test_station(model, test_dataset, savepath, Transform, Common.test_save_index)
        predictions.to('cpu')
        targets.to('cpu')
        # 计算指标
        index_values = CalculateIndex(predictions, targets, station)

        # 将每个站点的指标值添加到列表中
        all_hinge_loss.append(index_values['HingeLoss'])
        all_logistic_loss.append(index_values['LogisticLoss'])
        all_lambda_loss1.append(index_values['LambdaLoss1'])
        all_lambda_loss2.append(index_values['LambdaLoss2'])
        all_rmse.append(index_values['RMSE'])
        all_accuracy0.append(index_values['Accuracy0'])
        all_correct_count0.append(index_values['Correct_count0'])
        all_accuracy1.append(index_values['Accuracy1'])
        all_correct_count1.append(index_values['Correct_count1'])
        all_accuracy2.append(index_values['Accuracy2'])
        all_correct_count2.append(index_values['Correct_count2'])
        all_kurtosis.append(index_values['Kurtosis'])
        all_srcc.append(index_values['SRCC'])
        all_krcc.append(index_values['KRCC'])
        all_iqr.append(index_values['IQR'])
        all_R_squared.append(index_values['R_squared'])
        all_cv.append(index_values['CV'])

    all_correct_count0 = torch.tensor(all_correct_count0, dtype=torch.float32)
    all_correct_count1 = torch.tensor(all_correct_count1, dtype=torch.float32)
    all_correct_count2 = torch.tensor(all_correct_count2, dtype=torch.float32)

    # 计算平均值
    mean_hinge_loss = torch.tensor(all_hinge_loss).mean().item()
    mean_logistic_loss = torch.tensor(all_logistic_loss).mean().item()
    mean_lambda_loss1 = torch.tensor(all_lambda_loss1).mean().item()
    mean_lambda_loss2 = torch.tensor(all_lambda_loss2).mean().item()
    mean_rmse = torch.tensor(all_rmse).mean().item()
    mean_accuracy0 = torch.tensor(all_accuracy0).mean().item()
    mean_correct_count0 = all_correct_count0.clone().detach().mean().item()
    mean_accuracy1 = torch.tensor(all_accuracy1).mean().item()
    mean_correct_count1 = all_correct_count1.clone().detach().mean().item()
    mean_accuracy2 = torch.tensor(all_accuracy2).mean().item()
    mean_correct_count2 = all_correct_count2.clone().detach().mean().item()
    mean_kurtosis = torch.tensor(all_kurtosis).mean().item()
    mean_srcc = torch.tensor(all_srcc).mean().item()
    mean_krcc = torch.tensor(all_krcc).mean().item()
    mean_iqr = torch.tensor(all_iqr).mean().item()
    mean_R_squared = torch.tensor(all_R_squared).mean().item()
    mean_cv = torch.tensor(all_cv).mean().item()

    # 打印输出平均值
    print("Average Metrics:\n"
        f"Mean Hinge Loss: {mean_hinge_loss:.4f}\t"
        f"Mean Logistic Loss: {mean_logistic_loss:.4f}\t"
        f"Mean Lambda Loss 1: {mean_lambda_loss1:.4f}\t"
        f"Mean Lambda Loss 2: {mean_lambda_loss2:.4f}\n"
        f"Mean RMSE: {mean_rmse:.4f}\t"
        f"Mean KRCC: {mean_krcc:4f}\t"
        f"Mean SRCC: {mean_srcc:4f}\t"
        f"Mean Accuracy:{mean_accuracy0:.4f} ({mean_accuracy1:.4f}) ({mean_accuracy2:.4f})\t"
        f"Mean correct count:{mean_correct_count0} ({mean_correct_count1}) ({mean_correct_count2})\t"
        f"Mean Kurtosis: {mean_kurtosis:.4f}\t"
        f"Mean IQR: {mean_iqr:.4f}\t"
        f"Mean R_squared: {mean_R_squared:.4f}\t"
        f"Mean CV: {mean_cv:.4f}\n"
    )
    return mean_accuracy0, mean_accuracy1, mean_accuracy2


def test_station(model, test_dataset, savepath, transform, save_index=True):
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)


    with torch.no_grad():
        model.eval()
        for image_tensor, input_data, label in test_loader:
            input_data = input_data.to(Common.device)
            prediction = model(input_data)
    return prediction.cpu(), label


def rmse(predictions, targets):
    mse = torch.mean((predictions - targets)**2)
    rmse_value = torch.sqrt(mse)
    return rmse_value


def plot_ranking_errors_line(predictions, targets):
    targets = np.array(targets).ravel()
    predictions = np.array(predictions).ravel()

    # 计算排序误差
    errors = np.abs(predictions - targets)

    plt.plot(targets, errors, label='Sorting Errors', marker='o')

    # 在折线下方填充阴影色（如果需要对排序后的数据进行操作，请使用sorted_targets和sorted_errors）
    plt.fill_between(targets, 0, errors, alpha=0.3)

    # 由于我们没有对数据进行排序，填充阴影可能不会正确显示，如果不需要填充可以注释掉上面这行

    # 添加y=1的红色线条
    plt.axhline(y=1, color='red', linestyle='-', label='y=1 Line')

    # 设置X轴范围
    plt.xlim(1, 20)

    # 设置图形属性
    # plt.title('Line Plot of Sorting Errors between Predictions and Targets')
    plt.xlabel('Targets')
    plt.ylabel('Sorting Errors')
    plt.legend()

    # 如果需要设置Y轴的最大值，可以取消注释下面的代码行并调整值
    plt.ylim(0, 20)

    # 显示图形
    plt.show()


def compute_accuracy(true_ranking, predicted_ranking, tolerance):
    errors = torch.abs(true_ranking - predicted_ranking)

    # 判断排名误差是否在允许范围内
    correct_predictions = errors <= tolerance

    # 计算准确度
    accuracy = torch.mean(correct_predictions.float()).item()

    # 计算正确预测的数量
    correct_count = torch.sum(correct_predictions).item()

    return accuracy, correct_count

def kurtosis_coefficient(data):
    mean_val = torch.mean(data)
    centered_data = data - mean_val
    n = data.size(1)  # 获取数据的第二个维度的大小
    fourth_moment = torch.sum(centered_data**4) / n
    second_moment_squared = (torch.sum(centered_data**2) / n)**2
    kurt = fourth_moment / second_moment_squared - 3
    return kurt

def r_squared(predicted, target):
    mean_target = torch.mean(target)

    # 计算总平方和
    total_sum_of_squares = torch.sum((target - mean_target)**2)

    # 计算残差平方和
    residual_sum_of_squares = torch.sum((target - predicted)**2)

    # 计算 R 方值
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)

    return r2.item()  # 将结果转换为标量值

def calculate_krcc(predictions, targets):
    predictions_np = predictions.detach().cpu().numpy().ravel()
    targets_np = targets.detach().cpu().numpy().ravel()

    # 使用kendalltau函数计算KRCC
    krcc, _ = kendalltau(predictions_np, targets_np)

    return krcc

def calculate_srcc(predictions, targets):
    predictions_np = predictions.detach().cpu().numpy().ravel()
    targets_np = targets.detach().cpu().numpy().ravel()

    # 使用spearmanr函数计算SRCC
    srcc, _ = spearmanr(predictions_np, targets_np)

    return srcc

def calculate_iqr(data):
    data_np = data.cpu().numpy()  # 将PyTorch Tensor转换为NumPy数组
    q1 = np.percentile(data_np, 25)
    q3 = np.percentile(data_np, 75)
    iqr = q3 - q1
    return iqr.item()


def coefficient_of_variation(data):
    mean_value = torch.mean(data)
    std_dev = torch.std(data)

    cv = (std_dev / mean_value) * 100.0

    return cv.item()

def Test_record():
    model = VisNet()

if __name__ == '__main__':
    # 获取模型
    model = VisNet()
    # model_name = "2023-12-25-18-59-43/1797-PairwiseHingeLoss_40.pth"  # 测试集参照模型
    # model_name = "2024-02-02-10-45-57/1717-PairwiseHingeLoss_40.pth"  # 0.7833 (0.9208) W-Hingle（改）(基准) + 2000~0.5
    # model_name = "2024-02-26-21-29-13/111-LambdaARPLoss2_40.pth"
    # model_name = "2024-02-27-15-57-43/866-PairwiseHingeLoss_40.pth"
    # model_name = "2024-02-27-19-36-12/86-LambdaARPLoss2_40.pth"  # 0.6708 (0.7875)
    #model_name = "2024-02-29-09-42-51/100-LambdaARPLoss2_40.pth"  # 0.7042(0.8167) W ARPLoss
    #model_name = "2024-04-17-10-08-22/181-PairwiseHingeLoss_40.pth"  # 0.7458 (0.8708) ARP+Hingle
    # model_name = "2024-04-30-09-07-52/0.7167_490-PairwiseHingeLoss_40.pth"
    #model_name = "2024-11-08-09-41-09/0.741666615009307923-PairwiseHingeLoss_40.pth"  # 0.7417 (0.9042) No-W HingleLoss
    #model_name = "2024-11-08-09-57-14/0.68333333730697630-LambdaARPLoss2_40.pth"  # 0.6833 (0.8208) No_W ARPLoss
    # model_name = "2024-12-23-16-06-14/0.7833333611488342674-PairwiseHingeLoss_40.pth" #W-H+depth
    model_name = "2024-12-30-15-49-11/0.8041665554046631_1-Tensor_40.pth"  # W-H+depth

    modelPath = Train.model_pre_trained + model_name
    model.load_state_dict(torch.load(modelPath))
    model.to(Common.device)
    model.eval()
    Test(model, transform)



