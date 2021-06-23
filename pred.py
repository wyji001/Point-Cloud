import argparse
import os
import random
import numpy
import numpy as np
from torch import nn
import torch.optim as optim
import torch.utils.data
from pointmodel import get_model,  MyDataset
torch.set_default_tensor_type(torch.DoubleTensor)
from  pointtransform import PointTransformerCls, arg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file', type=str,required=True, help='input file')
    parser.add_argument(
        '--model', type=str, help='PointNet or PointTransformer', default=6)
    opt = parser.parse_args()
    if opt.model == "PointNet":
        model_list = os.listdir("./Model/PointNet（B）")
        model1 = get_model().cuda()
        model2 = get_model().cuda()
        model3 = get_model().cuda()
        model4 = get_model().cuda()
        model5 = get_model().cuda()
        a = model_list[0]
        b = model_list[1]
        c = model_list[2]
        d = model_list[3]
        e = model_list[4]
        model1.load_state_dict(torch.load(f"./Model/PointNet（B）/{a}"))
        model2.load_state_dict(torch.load(f"./Model/PointNet（B）/{b}"))
        model3.load_state_dict(torch.load(f"./Model/PointNet（B）/{c}"))
        model4.load_state_dict(torch.load(f"./Model/PointNet（B）/{d}"))
        model5.load_state_dict(torch.load(f"./Model/PointNet（B）/{e}"))
    elif opt.model == "PointTransformer":
        model_list = os.listdir("./Model/PointTransformer（B）")
        cls = arg()
        model1 = PointTransformerCls(cls).cuda()
        model2 = PointTransformerCls(cls).cuda()
        model3 = PointTransformerCls(cls).cuda()
        model4 = PointTransformerCls(cls).cuda()
        model5 = PointTransformerCls(cls).cuda()
        a = model_list[0]
        b = model_list[1]
        c = model_list[2]
        d = model_list[3]
        e = model_list[4]
        model1.load_state_dict(torch.load(f"./Model/PointTransformer（B）/{a}"))
        model2.load_state_dict(torch.load(f"./Model/PointTransformer（B）/{b}"))
        model3.load_state_dict(torch.load(f"./Model/PointTransformer（B）/{c}"))
        model4.load_state_dict(torch.load(f"./Model/PointTransformer（B）/{d}"))
        model5.load_state_dict(torch.load(f"./Model/PointTransformer（B）/{e}"))
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    all_pred_list = []
    for i in range(5):
        point_data = np.loadtxt(opt.file).astype(np.float64)
        theta1 = np.random.uniform(0, np.pi * 2)
        theta2 = np.random.uniform(0, np.pi * 2)
        theta3 = np.random.uniform(0, np.pi * 2)
        trans_matrix1 = np.array([[np.cos(theta1), -np.sin(theta1)],
                                  [np.sin(theta1), np.cos(theta1)]])
        trans_matrix2 = np.array([[np.cos(theta2), -np.sin(theta2)],
                                  [np.sin(theta2), np.cos(theta2)]])
        trans_matrix3 = np.array([[np.cos(theta3), -np.sin(theta3)],
                                  [np.sin(theta3), np.cos(theta3)]])
        point_data[:, [0, 1]] = point_data[:, [0, 1]].dot(trans_matrix1)
        point_data[:, [0, 2]] = point_data[:, [0, 2]].dot(trans_matrix2)
        point_data[:, [1, 2]] = point_data[:, [1, 2]].dot(trans_matrix3)
        point_data = torch.Tensor(np.expand_dims(point_data, axis=0)).cuda()
        if opt.model == "PointNet":
            point_data = point_data.transpose(2, 1)
        out1 = model1(point_data).cpu().detach().numpy()[0][0]
        out2 = model2(point_data).cpu().detach().numpy()[0][0]
        out3 = model3(point_data).cpu().detach().numpy()[0][0]
        out4 = model4(point_data).cpu().detach().numpy()[0][0]
        out5 = model5(point_data).cpu().detach().numpy()[0][0]
        all_pred_list.append(out1)
        all_pred_list.append(out2)
        all_pred_list.append(out3)
        all_pred_list.append(out4)
        all_pred_list.append(out5)
    all_pred_mean = np.array(all_pred_list).mean()
    print(f"Pred\t:\t{all_pred_mean}\n")







