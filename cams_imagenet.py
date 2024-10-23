import os
import argparse
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F

from gradcam import ISGCAM
from sklearn.metrics import accuracy_score, f1_score, recall_score

drop_stride = 2

class CAMs:
    
    def __init__(self, args):
        self.cam_method = args.cam_method.lower()
        self.classfier = args.classfier.lower()
        self.batch_size = args.batch_size
        self.topk = args.topk
        self.alpha = args.alpha
        self.m = args.m
        self.fix_ret = args.fix_ret
        
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.num_worker = args.num_workers if torch.cuda.is_available() else 0
        
        self.loss = nn.CrossEntropyLoss()
        
    def start(self):
        cams_dict = {'isgcam': ISGCAM}
        classfier_dict = {'vgg': models.vgg16(pretrained=True),
                        'squeezenet': models.squeezenet1_1(pretrained=True)}
        layer_name = {'vgg': 'features_29',
                    'squeezenet': 'features_12_expand3x3_activation'}
        
        self.model = classfier_dict[self.classfier]
        self.model.eval(), self.model.to(self.device)
        model_dict = dict(type=self.classfier, 
                        arch=self.model,
                        layer_name=layer_name[self.classfier], 
                        input_size=(224, 224))

        cam = cams_dict[self.cam_method]
        if self.cam_method == 'bicam':
            cam_for_model = cam(model_dict, True, self.alpha, self.m, self.fix_ret)
            print("alpha =", self.alpha, ", m =", self.m, ", fix_ret =", self.fix_ret)
        else:
            cam_for_model = cam(model_dict, True)
        
        out = self.CAM_imagenet(cam_for_model)
        
    def summary(self, ave_drop, incr, inse_auc, del_auc):
        if self.count == 0:
            self.ave_drop = ave_drop
            self.increase = incr
            self.inse_auc = inse_auc
            self.del_auc = del_auc

        else:
            self.ave_drop += ave_drop
            self.increase += incr
            self.inse_auc += inse_auc
            self.del_auc += del_auc

        self.count += 1

        if self.count % 50 == 0:
            # print("Cross Entropy:\n", self.loss_value)
            # for i in range(len(self.topk)):
            #     print("Top", self.topk[i], ":", self.top_k[:, i],
            #           "Batch size:", self.batch_size,
            #           "Count:", self.count,
            #           "Data count:", self.data_set_len)
            print("Average Drop:", self.ave_drop, 
                  ", Increase:", self.increase, 
                  ", Insersion AUC:", self.inse_auc, 
                  ", Deletion AUC:", self.del_auc,
                  ", Data count:", self.data_set_len)
            # print()

    def CAM_imagenet(self, cam):
        data_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        
        val_dataset = torchvision.datasets.ImageNet(root='./ImageNet2012/', split='val', transform=data_transform)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_worker)
        
        self.count = 0
        self.data_set_len = 0
            
        for data, label in val_loader:
            data, label = data.to(self.device), label.to(self.device)
            mask, _ = cam(data)
            ave_drop, incr, inse_auc, del_auc = self.evaluate(data, mask, label)
            self.data_set_len += data.size(0)
            self.summary(ave_drop, incr, inse_auc, del_auc)
            
            if self.data_set_len > 2005:
                print("End.")
                return
            


        self.loss_value /= self.data_set_len
        self.ave_drop /= self.data_set_len
        print("Global Mean Cross Entropy:\n", self.loss_value)

        for i in range(len(self.topk)):
            print("Top", self.topk[i], ":", self.top_k[:, i],
                    "Batch size:", self.batch_size,
                    "Data Count:", self.data_set_len)

        print("Global Average Drop:", self.ave_drop)
        print("Global Increase Num:", self.increase, "\nGlobal Data Num:", self.count)
        self.increase = self.increase.astype(float)
        self.increase /= self.data_set_len
        print("Global Increase Pro:", self.increase)
                             
    def evaluate(self, data, saliency_map, label):
        b, k, u, v = saliency_map.shape # k = 1 for saliency map
        saliency_map = F.relu(saliency_map)


        ## 1. ave drop ave increase
        threshold = np.percentile(saliency_map.cpu(), 50)
        drop_data = torch.where(saliency_map.repeat(1, 3, 1, 1) > threshold, 
                                data, torch.zeros_like(data).to(data.device))

        # calculate average drop and increase of confidence
        # drop_data = data.clone().detach()
        # drop_data = drop_data * saliency_map
        #3 mute 50% of the data
        # drop_data[indices.repeat(1, 3, 1, 1) > 25088] =  0.5 # 224 * 224 * 0.5 = 25088

        with torch.no_grad():
            self.model.zero_grad()
            model_output = self.model(data)
            model_output = F.softmax(model_output, dim=-1)

            self.model.zero_grad()
            drop_output = self.model(drop_data)
            drop_output = F.softmax(drop_output, dim=-1)

        one_hot = torch.zeros(model_output.shape, dtype=torch.float).to(self.device)
        model_class = model_output.argmax(dim=1, keepdim=True)
        one_hot = one_hot.scatter_(1, model_class, 1)

        score = torch.sum(one_hot * model_output, dim=1)
        drop_score = torch.sum(one_hot * drop_output, dim=1)

        average_drop = (F.relu(score - drop_score) / score).sum().detach().cpu().numpy()
        increase = (score < drop_score).sum().detach().cpu().numpy()

        ## 2. insertion  deletion
        pixel_num = 50175 # 224 * 224
        drop_num = 100 // drop_stride

        saliency_map = saliency_map.cpu()
        threshold = [np.percentile(saliency_map, i*drop_stride) for i in range(drop_num, 0, -1)]
        saliency_map = saliency_map.to(self.device).repeat(1, 3, 1, 1)
        # print(threshold)

        insersion_list = []
        for drop_radio in range(drop_num): 
            drop_data = torch.where(saliency_map > threshold[drop_radio], data,
                                torch.zeros_like(data).to(self.device))
            
            self.model.zero_grad()
            with torch.no_grad():
                drop_logit = self.model(drop_data)
                drop_pro = F.softmax(drop_logit, dim=-1)
            drop_score = torch.sum(one_hot * drop_pro, dim=1)
            insersion_list.append(drop_score)

        insersion_list = np.array(insersion_list)
        # print(insersion_list)
        insersion_auc = insersion_list.sum() * drop_stride / 100
        insersion_auc = insersion_auc.cpu().numpy()

        deletion_list = []
        for drop_radio in range(drop_num): 
            drop_data = torch.where(saliency_map > threshold[drop_radio], 
                                torch.zeros_like(data).to(self.device), data)
            
            self.model.zero_grad()
            with torch.no_grad():
                drop_logit = self.model(drop_data)
                drop_pro = F.softmax(drop_logit, dim=-1)
            drop_score = torch.sum(one_hot * drop_pro, dim=1)
            deletion_list.append(drop_score)

        deletion_list = np.array(deletion_list)
        deletion_auc = deletion_list.sum() * drop_stride / 100
        deletion_auc = deletion_auc.cpu().numpy()

        
        
        return average_drop, increase, insersion_auc, deletion_auc

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')
    parser.add_argument('--cam_method', type=str, default='ISGCAM', help='Input cam mothod')
    parser.add_argument('--classfier', type=str, default='vgg', help="one of vgg and squeezenet")
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--device', type=str, default="0", help="device")
    parser.add_argument('--num_workers', type=int, default=1, help="num of wrokers")
    parser.add_argument('--topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
    parser.add_argument('--alpha', type=float, default=1.0, help="alpha for bicam")
    parser.add_argument('--m', type=int, default=100, help="m for bicam")
    parser.add_argument('--fix_ret', action='store_true', default=False, help="fix ret for bicam")
    args = parser.parse_args()

    k = CAMs(args)
    k.start()