import torch
from torchvision import transforms, datasets, utils
import json, os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random

# 为 DataLoader 的每个工作线程设置随机种子
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# 使用 torch.Generator 确保数据加载顺序一致
g = torch.Generator()
g.manual_seed(42)
torch.random.manual_seed(g.initial_seed())

# 数据预处理，定义data_transform这个字典
data_transform = {
    "train": transforms.Compose([
                                 transforms.Resize((128, 128)),
                                 # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
                                 # transforms.RandomResizedCrop(128),
                                 transforms.ToTensor(),
                                 transforms.ColorJitter(hue=0.5),
                                 # transforms.RandomRotation((-45,45)),
                                 transforms.RandomHorizontalFlip(),
                                 # transforms.RandomRotation((-45, 45)),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225]),
                                 ]),

    "val": transforms.Compose([transforms.Resize((128, 128)),
                               # transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])])}

def train_data(root_dir,batch_size):
    train_dataset = datasets.ImageFolder(root=root_dir + "train",transform=data_transform["train"])
    train_data_num = len(train_dataset)
    # print(train_dataset.classes)
    flower_list = train_dataset.class_to_idx  # 获取分类的名称所对应的索引，即{'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    cla_dict = dict((val, key) for key, val in flower_list.items())  # 遍历获得的字典，将key和value反过来，即key变为0，val变为daisy
    # 将key和value反过来的目的是，预测之后返回的索引可以直接通过字典得到所属类别
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:  # 保存入json文件
        json_file.write(json_str)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,shuffle=True,
                                            num_workers=8)#8
    return train_loader,train_data_num

def val_data(root_dir,batch_size):
    validate_dataset = datasets.ImageFolder(root=root_dir + "val",transform=data_transform["val"])
    val_data_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=8)#8
    return validate_loader,val_data_num

def ImageFolder_dataloader(root_dir, bs, is_train=True, nw=16):
    # 根据是否训练模式来切换数据集路径和数据增强模式
    root = root_dir + ("train" if is_train else "val")
    transform = data_transform["train"] if is_train else data_transform["val"]
    # ImageFolder 获取数据
    ImageFolder_dataset = datasets.ImageFolder(root=root, transform=transform)
    data_len = len(ImageFolder_dataset)

    class_index = ImageFolder_dataset.class_to_idx  # 获取分类的名称所对应的索引，即{'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    index_class = dict((val, key) for key, val in class_index.items())  # 遍历获得的字典，将key和value反过来，即key变为0，val变为daisy
    # 将key和value反过来的目的是，预测之后返回的索引可以直接通过字典得到所属类别
    # write dict into json file
    json_str = json.dumps(index_class, indent=4)
    with open('class_indices.json', 'w') as json_file:  # 保存入json文件
        json_file.write(json_str)
    data_loader = torch.utils.data.DataLoader(ImageFolder_dataset,
                                            batch_size=bs,
                                            shuffle = True if is_train else False,
                                            num_workers=nw,
                                            worker_init_fn=worker_init_fn, generator=g
                                            )
    return data_loader, data_len





class FERDataset(Dataset):
    def __init__(self, dataset_path, is_train):
        super(FERDataset, self).__init__()
        self.dataset_path = dataset_path
        self.is_train = is_train
        self.root_dir = os.path.join(self.dataset_path, "train" if self.is_train else "val")
        # 类别索引映射关系
        class_to_index_dict = { "engraving":0,
                                "heat":     1,
                                "ink":      2,
                                "laser":    3,}
        # 初始化数据集文件路径
        self.images_list = []
        self.labels_list = []
        for class_name in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, class_name)
            # 获取该类别的索引
            class_id = class_to_index_dict[class_name]
            # print(type(class_id))
            # 遍历各类别
            for image_name in os.listdir(class_dir):
                # 获取图像路径
                image_path = os.path.join(class_dir, image_name)
                # 添加到列表中
                self.images_list.append(image_path)
                # 同时添加类别
                self.labels_list.append(class_id)

        self.img_trans = self.img_transformer()

    def __len__(self):
        return len(self.images_list)

    def img_transformer(self):
            return {"train": transforms.Compose([
                                 transforms.Resize((128, 128)),
                                 # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
                                 # transforms.RandomResizedCrop(128),
                                 transforms.ColorJitter(hue=0.5),
                                 # transforms.RandomRotation((-45,45)),
                                 transforms.RandomHorizontalFlip(),
                                 # transforms.RandomRotation((-45, 45)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225]),
                                 ]),

                    "val": transforms.Compose([transforms.Resize((128, 128)),
                               # transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])])}


    def __getitem__(self, index):
        # 读取原图像
        ori_img = Image.open(self.images_list[index]).convert('RGB')
        trans_img = self.img_trans["train"](ori_img) if self.is_train else self.img_trans["val"](ori_img)

        return trans_img, self.labels_list[index]

def Gen_DataLoader(data_path, is_train, bs=8):
    gen_dataset = FERDataset(dataset_path = data_path, is_train = is_train)
    data_len = gen_dataset.__len__()
    # print(gen_dataset.__len__())
    data_loader =  torch.utils.data.DataLoader(
                                gen_dataset,
                                batch_size=bs,
                                shuffle = True if is_train else False,
                                num_workers=16,
                                drop_last=False)
    return data_loader, data_len



if __name__== "__main__":

    # data_dir = "/datasets/rafdb/"
    data_dir = 'D:\\Data\\Research_data\\dataset\\FER\\RAFDB\\basic\\rafdb\\'
    batch_size = 2

    # Old ImageFolder 方式
    # validate_loader,val_data_num = val_data(data_dir,batch_size)
    # print("Total data:", val_data_num)
    # for data_batch in validate_loader:
    #          datas, labels = data_batch
    #          labels_len = len(labels)
    #          print(datas.size(), labels_len, type(labels))

    # ImageFolder 方式
    # data_loader, data_len = ImageFolder_dataloader(root_dir = data_dir, bs = batch_size, is_train=False)
    # print("Total data:", data_len)
    # for data_batch in data_loader:
    #          datas, labels = data_batch
    #          labels_len = len(labels)
    #          # print(datas.size(), labels_len, type(labels))

    # # Dataset
    data_loader, data_len = Gen_DataLoader(data_path=data_dir, is_train=False, bs=batch_size)
    print("Total data:", data_len)
    for step, data_batch in enumerate(data_loader, start=0):
        datas, labels = data_batch
        labels_len = len(labels)
        print(datas.size(), labels_len, type(labels))
 