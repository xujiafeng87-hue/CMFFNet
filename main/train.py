import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import torchvision.models as models
import numpy as np
import torch.optim as optim
import sys
from torch.nn import functional as F
import os, sys
import json
import time, random
from tensorboardX import SummaryWriter

import data_input


# sys.path.append('./')

# 控制seed
def seed_torch(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # 关闭自动选择最快算法的选项
    torch.backends.cudnn.deterministic = True  # 让 CuDNN 以确定性的方式执行


seed_torch()

# 执行控制seed种子之后再倒入代码包
import data_input
# from model import ResNet18
from MedMamba import VSSM as medmamba


def cre_prolog():
    # 获取当前脚本的绝对路径
    current_script_path = os.path.abspath(__file__)
    # 获取当前脚本的所在目录
    parent_dir = os.path.dirname(current_script_path)
    # 获取更上一级目录（即父目录的父目录）
    grandparent_dir = os.path.dirname(parent_dir)

    # 定义要创建的Pro_log目录的路径（在更上一级目录中）
    pro_log_dir_path = os.path.join(grandparent_dir, 'Pro_log_5_10')

    # 检查Pro_log目录是否存在，如果不存在则创建
    if not os.path.exists(pro_log_dir_path):
        os.makedirs(pro_log_dir_path)
        print(f"Directory '{pro_log_dir_path}' created.")

        # 获取当前脚本的上一级目录的名称（用于新子目录的名称）
    parent_dir_name = os.path.basename(parent_dir)
    # 在Pro_log目录中创建以当前脚本的上一级目录名为名字的目录
    new_subdir_path = os.path.join(pro_log_dir_path, parent_dir_name)
    # 检查这个新目录是否存在，如果不存在则创建
    if not os.path.exists(new_subdir_path):
        os.makedirs(new_subdir_path)
        print(f"Directory '{new_subdir_path}' created.")
    else:
        print(f"Directory '{new_subdir_path}' already exists.")

    return new_subdir_path


# 整个训练脚本产生的结果都将存入这个脚本中，模型，log等等
exp_dir = cre_prolog()
print(f"实验目录 {exp_dir}")

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")  # 运行时改
print(device)


class LSR(nn.Module):
    def __init__(self, n_classes=4, eps=0.1):
        super(LSR, self).__init__()
        self.n_classes = n_classes
        self.eps = eps

    def forward(self, outputs, labels):
        # labels.shape: [b,]
        assert outputs.size(0) == labels.size(0)
        n_classes = self.n_classes
        one_hot = F.one_hot(labels, n_classes).float()
        mask = ~(one_hot > 0)
        smooth_labels = torch.masked_fill(one_hot, mask, self.eps / (n_classes - 1))
        smooth_labels = torch.masked_fill(smooth_labels, ~mask, 1 - self.eps)
        # ce_loss = torch.sum(-smooth_labels * F.log_softmax(outputs, 1), dim=1).mean() # 标准是用这个
        ce_loss = torch.sum(-smooth_labels * F.log_softmax(outputs, 1), dim=1)
        # print(ce_loss/ce_loss.mean())
        # ce_loss = F.nll_loss(F.log_softmax(outputs, 1), labels, reduction='mean')
        return ce_loss


LSR_loss = LSR(n_classes=4, eps=0.1)

# data_dir = "/datasets/rafdb/"
data_dir = "/home/ailab/hlj/Passport/数据集2025-12/"  # 运行时改为总数据集路径＋/
batch_size = 32
# trainset loader
train_loader, tra_num = data_input.ImageFolder_dataloader(data_dir, bs=batch_size, is_train=True, nw=8)
# print(train_loader)
# validation loader
validate_loader, val_num = data_input.ImageFolder_dataloader(data_dir, bs=64, is_train=False, nw=4)

net = medmamba(
    depths=[2, 2, 4, 2],
    dims=[96, 192, 384, 768],
    num_classes=4
)

# 预训练权重，用于恢复训练
pre_train_model = 'AlexNet3_vit.pth'
# save_path = './AlexNet3_vit2.pth'

is_recovery = False
if is_recovery:
    checkpoint = torch.load(pre_train_model)
    net.load_state_dict(checkpoint)
    print("-------- 模型恢复训练成功-----------")
    save_path = './pretrain_{}_.pth'.format(pre_train_model.split('.')[0])

net.to(device)
# weght_ce = torch.tensor([0.0939, 0.4310, 0.1689, 0.0254, 0.0611, 0.1718, 0.0480], dtype=torch.float32).to(device)
# weight=weght_ce
loss_function = nn.CrossEntropyLoss()
pata = list(net.parameters())  # 查看net内的参数 lr 0.0001
optimizer = optim.Adam(net.parameters(), lr=0.00001, weight_decay=1e-4)

# scheduler= torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 80, eta_min=1e-6, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
import random

best_acc = 0.0

# 在代码开始运行时删除旧的日志文件
log_file = f"{exp_dir}/log.txt"
if os.path.exists(log_file):
    os.remove(log_file)

# === ADD === 创建用于记录每 epoch 指标的文本文件（格式化对齐）
metrics_txt = f"{exp_dir}/epoch_metrics.txt"
with open(metrics_txt, 'w') as f:
    # 写入表头，用制表符或空格对齐（这里用制表符 \t 更通用）
    f.write("Epoch\tTrain_Acc\tTrain_Loss\tVal_Acc\tone_epoch_time\n")

# === ADD === 初始化 TensorBoard SummaryWriter
writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'tensorboard'))

# 统计训练总时间
t0 = time.perf_counter()
for epoch in range(90):
    # train
    net.train()  # 在训练过程中调用dropout方法
    tra_loss = 0.0
    t1 = time.perf_counter()  # 统计训练一个epoch所需时间
    # print('star')
    tra_acc = 0.0
    # torch.autograd.set_detect_anomaly(True)
    for step, data in enumerate(train_loader, start=0):
        # 这个时候的标签还是数字 不是onehot
        # print(step)
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss_list = LSR_loss(outputs, labels)
        loss = loss_list.mean()
        # 累加每个步骤的损失
        tra_loss += loss_list.sum()
        loss.backward()
        optimizer.step()
        tra_predict_y = torch.max(outputs, dim=1)[1]
        step_acc = (tra_predict_y == labels.to(device)).sum().item()
        tra_acc += step_acc
        # each 100 step(or batch) print once
        if (step + 1) % 100 == 0:
            print("step:{} train acc:{:.5f} train loss:{:.5f}".format(step, step_acc / len(labels), loss))

    scheduler.step()
    one_epoch_time = time.perf_counter() - t1

    # 计算并打印整个epoch的平均损失和准确率
    tra_loss = tra_loss / tra_num
    tra_acc = tra_acc / tra_num

    # validate
    # ema.apply_shadow()
    net.eval()  # 在测试过程中关掉dropout方法，不希望在测试过程中使用dropout
    # ema.restore()
    val_acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():

        for data_test in validate_loader:
            test_images, test_labels = data_test
            test_labels_len = len(test_labels)
            outputs = net(test_images.to(device))
            # out= outputs
            predict_y = torch.max(outputs, dim=1)[1]
            val_acc += (predict_y == test_labels.to(device)).sum().item()

        val_acc = val_acc / val_num
        torch.save(net.state_dict(), f"{exp_dir}/current_model.pth")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), f"{exp_dir}/best_model.pth")
            if 0.91 < val_acc:
                torch.save(net.state_dict(), f"{exp_dir}/{val_acc:.5f}_model.pth")

        log = '\n[epoch %d] tra_acc:%.5f tra_loss: %.5f  test_acc: %.5f best_acc: %.5f epoch_time:%.5f' % \
              (epoch + 1,
               tra_acc,
               tra_loss,
               val_acc,
               best_acc,
               one_epoch_time)

        print(log)
        with open(log_file, 'a') as file:
            file.write(log)
        # === ADD === 将当前 epoch 的指标写入格式化文本文件(本文件只用于画loss和val_acc的图)
        with open(metrics_txt, 'a') as f:
            f.write(f"{epoch + 1}\t{tra_acc:.5f}\t{tra_loss:.5f}\t{val_acc:.5f}\t{one_epoch_time:.5f}\n")

        # === ADD === 将指标写入 TensorBoard
        writer.add_scalar('Accuracy/train', tra_acc, epoch + 1)
        writer.add_scalar('Accuracy/val', val_acc, epoch + 1)
        writer.add_scalar('Loss/train', tra_loss, epoch + 1)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch + 1)

# === ADD === 关闭 TensorBoard writer
writer.close()

train_time = time.perf_counter() - t0
print(f'Finished Training {train_time:.3f}s, about {train_time / 60:.4f}min')
