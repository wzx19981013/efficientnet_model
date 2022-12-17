from __future__ import print_function, division
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os

from efficientnet_pytorch import EfficientNet

# some parameters
use_gpu = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data_dir = 'data'
batch_size = 1
lr = 0.01
momentum = 0.9
num_epochs = 100
input_size = 224
class_num = 6
net_name = 'efficientnet-b0'


def loaddata(data_dir, batch_size, set_name, shuffle):
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [set_name]}
    # num_workers=0 if CPU else =1
    dataset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                      batch_size=batch_size,
                                                      shuffle=shuffle, num_workers=1) for x in [set_name]}
    data_set_sizes = len(image_datasets[set_name])
    with open("classToIndex.txt", "w") as f2:
        for key, value in image_datasets['test'].class_to_idx.items():
            f2.write(str(key) + " " + str(value) + '\n')
    # print(image_datasets['test'].imgs[0][0])
    return dataset_loaders, data_set_sizes


def test_model(model, criterion):
    result = pd.DataFrame(columns=['img_name', 'label', 'pred_label'])
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    outPre = []
    outLabel = []
    pred_list = []
    dset_loaders, dset_sizes = loaddata(data_dir=data_dir, batch_size=2, set_name='test', shuffle=False)
    # print(train_dataset.class_to_idx) #按顺序为这些类别定义索引为0,1...
    for data in dset_loaders['test']:
        inputs, labels = data
        labels = torch.squeeze(labels.type(torch.LongTensor))
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        if cont == 0:
            outPre = outputs.data.cpu()
            outLabel = labels.data.cpu()
        else:
            outPre = torch.cat((outPre, outputs.data.cpu()), 0)
            outLabel = torch.cat((outLabel, labels.data.cpu()), 0)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        pred_list.append(preds)
        pred_label = torch.cat(pred_list, dim=0)
        cont += 1
        # print("label",outLabel)
    print('Loss: {:.4f} Acc: {:.4f}'.format(running_loss / dset_sizes,
                                            running_corrects.double() / dset_sizes))
    result['label'] = outLabel
    result['pred_label'] = pred_label.data.cpu()
    img_list = []
    for i in range(dset_sizes):
         img_list.append(dset_loaders['test'].dataset.imgs[i][0])
    result['img_name'] = img_list
    result.to_csv("./result.csv",columns=['img_name','label','pred_label'], index=0)

def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=10):
    """Decay learning rate by a f#            model_out_path ="./model/W_epoch_{}.pth".format(epoch)
#            torch.save(model_W, model_out_path) actor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.8**(epoch // lr_decay_epoch))
    print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

if __name__ == '__main__':
    # test
    model_ft = EfficientNet.from_name(net_name)
    # 修改全连接层
    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, class_num)

    criterion = nn.CrossEntropyLoss()
    if use_gpu:
        model_ft = model_ft.cuda()
        criterion = criterion.cuda()

    optimizer = optim.SGD((model_ft.parameters()), lr=lr,
                          momentum=momentum, weight_decay=0.0004)

    # train_loss, best_model_wts = train_model(model_ft, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)
    best_model_path = ".\data\model\efficientnet-b0.pth"
    # test
    print('-' * 10)
    print('Test Accuracy:')
    model_ft = torch.load(best_model_path)
    # model_ft.load_state_dict(best_model_path)
    criterion = nn.CrossEntropyLoss().cuda()
    test_model(model_ft, criterion)
