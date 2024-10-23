import torch
import torch.nn.functional as F

from utils import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, find_squeezenet_layer


class GradCAM(object):
    """Calculate GradCAM salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcam = GradCAM(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """
    def __init__(self, model_dict, verbose=False):
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        if 'vgg' in model_type.lower():
            target_layer = find_vgg_layer(self.model_arch, layer_name)
        elif 'resnet' in model_type.lower():
            target_layer = find_resnet_layer(self.model_arch, layer_name)
        elif 'densenet' in model_type.lower():
            target_layer = find_densenet_layer(self.model_arch, layer_name)
        elif 'alexnet' in model_type.lower():
            target_layer = find_alexnet_layer(self.model_arch, layer_name)
        elif 'squeezenet' in model_type.lower():
            target_layer = find_squeezenet_layer(self.model_arch, layer_name)

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print("please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
                pass
            else:
                device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
                self.model_arch(torch.zeros(1, 3, *(input_size), device=device))
                print('saliency_map size :', self.activations['value'].shape[2:])


    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        if class_idx is None:
            class_idx = logit.argmax(dim=1, keepdim=True)

        one_hot = torch.zeros_like(logit).to(logit.device)
        one_hot = one_hot.scatter_(1, class_idx, 1)
        score = torch.sum(one_hot * logit)
        class_num = logit.size(1)

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        #alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map.detach(), logit.detach()

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)


class GradCAMpp(GradCAM):
    
    def __init__(self, model_dict, verbose=False):
        super(GradCAMpp, self).__init__(model_dict, verbose)

    def forward(self, input, class_idx=None, retain_graph=False):
        
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        if class_idx is None:
            class_idx = logit.argmax(dim=1, keepdim=True)

        one_hot = torch.zeros_like(logit).to(logit.device)
        one_hot = one_hot.scatter_(1, class_idx, 1)
        score = torch.sum(one_hot * logit, dim=-1).view(-1, 1)
        class_num = logit.size(1)
            
        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value'] # dS/dA
        activations = self.activations['value'] # A
        b, k, u, v = gradients.size()

        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
                activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom+1e-7)
        positive_gradients = F.relu(score.exp()*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha*positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map-saliency_map_min).div(saliency_map_max-saliency_map_min).data

        return saliency_map.detach(), logit.detach()

class ScoreCAM(GradCAM):
    
    def __init__(self, model_dict, verbose=False):
        super(ScoreCAM, self).__init__(model_dict, verbose)

    def forward(self, input, class_idx=None, retain_graph=False):
        
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        if class_idx is None:
            class_idx = logit.argmax(dim=1, keepdim=True)

        one_hot = torch.zeros_like(logit).to(logit.device)
        one_hot = one_hot.scatter_(1, class_idx, 1)
        score = torch.sum(one_hot * logit, dim=-1).view(-1, 1)
        class_num = logit.size(1)
            
        self.model_arch.zero_grad()
        # score.backward(retain_graph=retain_graph)
        # gradients = self.gradients['value'] # dS/dA
        activations = self.activations['value'] # A
        b, k, u, v = activations.size()

        score_saliency_map = torch.zeros(b, 1, 224, 224).to(activations.device)
        self.model_arch.zero_grad()
        with torch.no_grad():
            for i in range(k):
                saliency_map = torch.unsqueeze(activations[:, i, :, :], 1).detach()
                saliency_map = F.interpolate(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
                # norm
                norm_saliency_map = data_norm(saliency_map)
                sub_logit = self.model_arch(input * norm_saliency_map)
                sub_logit = F.softmax(sub_logit, dim=1)
                sub_score = torch.sum(one_hot * sub_logit, dim=1).view(-1, 1, 1, 1)
                score_saliency_map += (sub_score * norm_saliency_map).detach()
        
        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map = data_norm(score_saliency_map)
        return score_saliency_map.detach(), logit.detach()

def channel_norm(data):
    b, k, u, v = data.shape
    data_min = data.view(b, k, -1).min(dim=-1, keepdim=True).values.view(b, k, 1, 1)
    data_max = data.view(b, k, -1).max(dim=-1, keepdim=True).values.view(b, k, 1, 1)
    denominator = torch.where(
        (data_max-data_min)!=0., data_max-data_min, torch.tensor(1.))

    return (data - data_min) / denominator

def data_norm(data):
    b, k, u, v = data.shape
    data_min = data.view(b, -1).min(dim=-1, keepdim=True).values.view(b, 1, 1, 1)
    data_max = data.view(b, -1).max(dim=-1, keepdim=True).values.view(b, 1, 1, 1)
    denominator = torch.where(
        (data_max-data_min)!=0., data_max-data_min, torch.tensor(1.).to(data.device))

    return (data - data_min) / denominator

def slope_norm(slope):
    b, k = slope.shape
    slope_min = slope.min(dim=-1, keepdim=True).values
    slope_max = slope.max(dim=-1, keepdim=True).values
    denominator = torch.where(
        (slope_max-slope_min)!=0., slope_max-slope_min, torch.tensor(1.).to(slope.device))
    
    return (slope - slope_min) / denominator

def max_data(a, b):
    return torch.where(a > b, a, b)

def resi_data(a, b):
    return F.relu(torch.where(a > b, a, b) - a)

class EFCAM(GradCAM):
    
    def __init__(self, model_dict, verbose=False):
        super(EFCAM, self).__init__(model_dict, verbose)

    def forward(self, input, class_idx=None, retain_graph=False):
        
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        logit = F.softmax(logit, dim=1)
        if class_idx is None:
            class_idx = logit.argmax(dim=1, keepdim=True)
        # logit = - torch.log(logit)

        one_hot = torch.zeros_like(logit).to(logit.device)
        one_hot = one_hot.scatter_(1, class_idx, 1)
        score = torch.sum(one_hot * logit, dim=-1).view(-1, 1)
        class_num = logit.size(1)
        # score = logit.sum(1)
        # print("score:",score)
            
        self.model_arch.zero_grad()
        # score.backward(retain_graph=retain_graph)
        # gradients = self.gradients['value'] # dS/dA
        activations = self.activations['value'] # A
        b, k, u, v = activations.size()
        
        slope = torch.zeros(b, k).to(activations.device)
        self.model_arch.zero_grad()
        with torch.no_grad():
            for i in range(k):
                saliency_map = torch.unsqueeze(activations[:, i, :, :], 1).detach()
                saliency_map = F.interpolate(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
                # norm
                norm_saliency_map = data_norm(saliency_map)
                norm_saliency_map = 1 - norm_saliency_map
                # norm_saliency_map[norm_saliency_map < 0.5] = 0 
                sub_logit = self.model_arch(input * norm_saliency_map)
                sub_logit = F.softmax(sub_logit, dim=1)
                # sub_logit = - torch.log(sub_logit)
                sub_score = torch.sum(one_hot * sub_logit, dim=1).view(-1, 1) # 这里使用原始模型的输出类别
                # sub_score = sub_logit.sum(1)
                slope[:, i:i+1] = (score - sub_score)
        # slope = F.softmax(slope, dim=1).view(b, k, 1, 1)
        slope = F.relu(slope)
        slope = slope_norm(slope)
        slope = slope.view(b, k, 1, 1)
        # print(slope)
        saliency_map = (slope * activations).sum(1, keepdim=True)
        score_saliency_map = F.relu(saliency_map)
        # 归一化
        score_saliency_map = data_norm(score_saliency_map)
        score_saliency_map = F.interpolate(score_saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
        return score_saliency_map.detach(), logit.detach()
    
# class EFCAM_margin(GradCAM):
    
#     def __init__(self, model_dict, verbose=False):
#         super(EFCAM_margin, self).__init__(model_dict, verbose)

#     def forward(self, input, class_idx=None, retain_graph=False):
        
#         b, c, h, w = input.size()
        
#         logit = self.model_arch(input)
#         logit = F.softmax(logit, dim=-1)
#         if class_idx is None:
#             class_idx = logit.argmax(dim=1, keepdim=True)

#         logit = - torch.log(logit)
#         one_hot = torch.zeros_like(logit).to(logit.device)
#         one_hot = one_hot.scatter_(1, class_idx, 1)
#         score = torch.sum(one_hot * logit, dim=-1).view(-1, 1)
#         # score = - score * torch.log(score) - (1-score) * torch.log(1-score)
        
#         self.model_arch.zero_grad()
#         activations = self.activations['value'] # A
#         b, k, u, v = activations.size()
        
#         slope = torch.zeros(b, k).to(activations.device)
#         self.model_arch.zero_grad()
#         with torch.no_grad():
#             for i in range(k):
#                 saliency_map = torch.unsqueeze(activations[:, i, :, :], 1).detach()
#                 saliency_map = F.interpolate(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
#                 # norm
#                 norm_saliency_map = data_norm(saliency_map)
#                 ret = norm_saliency_map.mean(dim=(-2, -1), keepdim=False)
#                 drop_data = torch.where(norm_saliency_map.repeat(1, 3, 1, 1) < (ret), 
#                                         torch.zeros_like(input), input)
#                 # norm_saliency_map = 1 - norm_saliency_map
#                 # norm_saliency_map[norm_saliency_map < 0.5] = 0 
#                 sub_logit = self.model_arch(drop_data)
#                 sub_logit = F.softmax(sub_logit, dim=1)
#                 sub_logit = - torch.log(sub_logit)
#                 sub_score = torch.sum(one_hot * sub_logit, dim=1, keepdim=False) # 这里使用原始模型的输出类别
#                 # sub_score = - sub_score * torch.log(sub_score) - (1-sub_score) * torch.log(1-sub_score)
#                 slope[:, i:i+1] = (score - sub_score)
#         slope = F.softmax(slope, dim=-1)
        
#         slope_s = torch.zeros_like(slope) #(b, k).to(activations.device)
#         sorted_slope, indices_slope = torch.sort(slope, dim=-1, descending=True)
        
#         sorted_activations = torch.zeros_like(activations)
#         for i in range(sorted_activations.size(0)):
#             sorted_activations[i, :, :, :] = activations[i, indices_slope[i, :], :, :]

#         m = k
        
#         mul_activation = torch.zeros_like(activations).to(input.device)
#         with torch.no_grad():
#             norm_saliency_map_a = torch.zeros_like(activations[:, 0:1, :, :])
#             for i in range(m):
#                 print(i)
#                 saliency_map_b = sorted_activations[:, i:i+1, :, :].detach()

#                 # 归一化
#                 norm_saliency_map_b = data_norm(saliency_map_b)
#                 res_saliency = data_norm(max_data(norm_saliency_map_a, norm_saliency_map_b))
#                 res_saliency_map = F.interpolate(res_saliency, size=(224, 224), mode='bilinear', align_corners=False)

#                 ret = res_saliency_map.mean(dim=(-2, -1), keepdim=False)
#                 res_data = torch.where(res_saliency_map.repeat(1, 3, 1, 1) < (ret), 
#                                         torch.zeros_like(input), input)
#                 # mask = torch.where(res_saliency_map < (ret), 
#                 #                     torch.zeros_like(res_saliency_map), torch.ones_like(res_saliency_map))

#                 # res_data = res_saliency_map * input
#                 self.model_arch.zero_grad()
#                 output = self.model_arch(res_data)
#                 output = F.softmax(output, dim=-1)
#                 output = - torch.log(output)
#                 score_res = torch.sum(one_hot * output, dim=1, keepdim=False)

#                 if i > 0:
#                     slope_s[:, i:i+1] = score - score_res # slope_s[:, i-1:i] - score_res #/ mask.sum(dim=(-2, -1), keepdim=False)
#                 else:
#                     slope_s[:, i:i+1] = score - score_res # 6.90 - score_res #/ mask.sum(dim=(-2, -1), keepdim=False) 1000分类
#                 norm_saliency_map_a = data_norm(max_data(norm_saliency_map_a, norm_saliency_map_b))
#                 mul_activation[:, i:i+1, :, :] = norm_saliency_map_a
        
#         slope_ss = torch.where(slope_s < 0, torch.zeros_like(slope_s), slope_s)
#         print(mul_activation.shape)
#         print(score)
#         print(slope_s)
#         slope_ss = F.softmax(slope_ss, dim=-1)
        
#         saliency_map = (sorted_slope.view(b, k, 1, 1) * mul_activation).sum(1, keepdim=True)
#         # 归一化
#         score_saliency_map = data_norm(F.relu(saliency_map))
#         score_saliency_map = F.interpolate(score_saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
#         return score_saliency_map.detach(), logit.detach(), mul_activation
'''
class EFCAM_margin(GradCAM):
    
    def __init__(self, model_dict, verbose=False):
        super(EFCAM_margin, self).__init__(model_dict, verbose)

    def forward(self, input, class_idx=None, retain_graph=False):
        
        b, c, h, w = input.size()
        
        logit = self.model_arch(input)
        logit = F.softmax(logit, dim=-1)
        if class_idx is None:
            class_idx = logit.argmax(dim=1, keepdim=True)

        logit = - torch.log(logit)
        one_hot = torch.zeros_like(logit).to(logit.device)
        one_hot = one_hot.scatter_(1, class_idx, 1)
        score = torch.sum(one_hot * logit, dim=-1).view(-1, 1)
        
        # self.model_arch.zero_grad()
        # score.backward(retain_graph=retain_graph)
        # gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = activations.size()
        
        slope = torch.zeros(b, k).to(activations.device)
        act = torch.zeros_like(activations)
        with torch.no_grad():
            for i in range(k):
                saliency_map = activations[:, i:i+1, :, :].detach()
                saliency_map = F.interpolate(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
                # norm
                norm_saliency_map = data_norm(saliency_map)

                ret = norm_saliency_map.mean(dim=(-2, -1), keepdim=False)
                centre_data = torch.where(norm_saliency_map.repeat(1, 3, 1, 1) <= ret, torch.zeros_like(input), input)
                surround_data = torch.where(norm_saliency_map.repeat(1, 3, 1, 1) > (ret), torch.zeros_like(input), input)
                # norm_saliency_map = 1 - norm_saliency_map
                # norm_saliency_map[norm_saliency_map < 0.5] = 0

                cen_logit = self.model_arch(centre_data)
                cen_logit = F.softmax(cen_logit, dim=1)
                cen_logit = - torch.log(cen_logit)
                cen_score = torch.sum(one_hot * cen_logit, dim=1, keepdim=False)

                # sur_logit = self.model_arch(surround_data)
                # sur_logit = F.softmax(sur_logit, dim=1)
                # sur_logit = - torch.log(sur_logit)
                # sur_score = torch.sum(one_hot * sur_logit, dim=1, keepdim=False)
                
                slope[:, i:i+1] = score - cen_score

        # slope = slope_norm(slope)
        slope = F.softmax(slope, dim=-1)
        # slope = torch.where(slope>0, slope_, torch.zeros_like(slope_))
        
        slope_s = torch.zeros_like(slope) #(b, k).to(activations.device)
        sorted_slope, indices_slope = torch.sort(slope, dim=-1, descending=True)
        
        sorted_activations = torch.zeros_like(activations)
        for i in range(sorted_activations.size(0)):
            sorted_activations[i, :, :, :] = activations[i, indices_slope[i, :], :, :]

        m = k
        
        mul_activation = torch.zeros(b, k, 224, 224).to(input.device)
        with torch.no_grad():
            norm_saliency_map_a = torch.zeros_like(input[:, 0:1, :, :])
            mask_a = torch.zeros_like(input[:, 0:1, :, :])
            # radio = torch.zeros_like(slope_s)
            for i in range(m):
                saliency_map_b = sorted_activations[:, i:i+1, :, :].detach()
                # 归一化
                norm_saliency_map_b = data_norm(saliency_map_b)
                norm_saliency_map_b = F.interpolate(norm_saliency_map_b, size=(224, 224), mode='bilinear', align_corners=False)
                ret_b = norm_saliency_map_b.mean(dim=(-2, -1), keepdim=False)
                mask_b = torch.where(norm_saliency_map_b <= ret_b, 
                                     torch.tensor(0.).to(input.device), 
                                     torch.tensor(1.).to(input.device))
                full_mask = max_data(mask_a, mask_b)
                resi_mask = resi_data(mask_a, mask_b)
                full_area = full_mask.sum(dim=(-2, -1), keepdim=False)
                resi_area = resi_mask.sum(dim=(-2, -1), keepdim=False)
                # b_area = mask_b.sum(dim=(-2, -1), keepdim=False)
                area_radio = resi_area / full_area

                res_data = (full_mask*0.5 + resi_mask*0.5).repeat(1, 3, 1, 1) * input
                self.model_arch.zero_grad()
                output = self.model_arch(res_data) * area_radio
                output = F.softmax(output, dim=-1)
                output = - torch.log(output)
                score_res = torch.sum(one_hot * output, dim=1, keepdim=False)

                slope_s[:, i:i+1] = (score - score_res)
                # radio[:, i:i+1] = area_radio
                # if i == 0:
                #     slope_s[:, i:i+1] = score - score_res
                #     store = slope_s[:, i:i+1]
                # else:
                #     slope_s[:, i:i+1] = score - score_res - store
                #     store = score - score_res
                
                mul_activation[:, i:i+1, :, :] = data_norm((1-area_radio)*norm_saliency_map_a + area_radio*norm_saliency_map_b)
                norm_saliency_map_a = mul_activation[:, i:i+1, :, :] #data_norm(norm_saliency_map_a * mask_a + norm_saliency_map_b * mask_b)
                mask_a = max_data(mask_a, mask_b)
        
        # print(score)
        # print(slope_s)
        slope_s = F.softmax(slope_s, dim=-1)# * radio
        # slope_s = slope_norm(slope_s)
        # slope_ss = torch.where(slope_ss <= 0, torch.zeros_like(slope_ss), slope_sss)
        # print(slope_ss)
        # ind = slope.argmax(-1, keepdim=False)
        # saliency_map = mul_activation[:, ind:ind+1, :, :]
        saliency_map = (slope_s.view(b, k, 1, 1) * mul_activation).sum(1, keepdim=True)
        # 归一化
        score_saliency_map = data_norm(F.relu(saliency_map))
        score_saliency_map = F.interpolate(score_saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
        return score_saliency_map.detach(), logit.detach(), mul_activation.detach()
'''

class EFCAM_margin(GradCAM):
    
    def __init__(self, model_dict, verbose=False):
        super(EFCAM_margin, self).__init__(model_dict, verbose)

    def forward(self, input, class_idx=None, retain_graph=False):
        
        b, c, h, w = input.size()
        
        logit = self.model_arch(input)
        logit = F.softmax(logit, dim=-1)

        if class_idx is None:
            class_idx = logit.argmax(dim=1, keepdim=True)
        # class_idx = torch.tensor([[282]]).to(logit.device)

        logit = - torch.log(logit)
        one_hot = torch.zeros_like(logit).to(logit.device)
        one_hot = one_hot.scatter_(1, class_idx, 1)
        score = torch.sum(one_hot * logit, dim=-1).view(-1, 1)
        
        # self.model_arch.zero_grad()
        # score.backward(retain_graph=retain_graph)
        # gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = activations.size()
        
        slope = torch.zeros(b, k).to(activations.device)
        act = torch.zeros_like(activations)
        with torch.no_grad():
            for i in range(k):
                saliency_map = activations[:, i:i+1, :, :].detach()
                saliency_map = F.interpolate(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
                # norm
                norm_saliency_map = data_norm(saliency_map)

                ret = norm_saliency_map.mean(dim=(-2, -1), keepdim=False)
                centre_data = torch.where(norm_saliency_map.repeat(1, 3, 1, 1) <= ret, torch.zeros_like(input), input)
                surround_data = torch.where(norm_saliency_map.repeat(1, 3, 1, 1) > ret, torch.zeros_like(input), input)
                mask = torch.where(norm_saliency_map.repeat(1, 3, 1, 1) <= ret, torch.zeros_like(input), torch.ones_like(input))
                # norm_saliency_map = 1 - norm_saliency_map
                # norm_saliency_map[norm_saliency_map < 0.5] = 0

                '''
                # 计算采样
                centre_numpy = centre_data.cpu().numpy()
                centre_map = torch.zeros_like(input)
                # for i in range(3):
                #     cen_hist, cen_bin = np.histogram(centre_numpy[:,i:i+1,:,:], bins=np.arange(256), density=True)
                #     # print("hist size:", cen_hist.size, cen_bin, centre_numpy[:,i:i+1,:,:].shape)
                #     cen_bin = cen_bin[1:-1] # 去除0和256
                #     cen_hist += cen_hist[0] / 254
                #     cen_hist = cen_hist[1:]
                #     same_distri_map = np.random.choice(cen_bin, size=centre_numpy[:,i:i+1,:,:].shape, replace=True, p=cen_hist)
                #     centre_map[:,i:i+1,:,:] = torch.from_numpy(same_distri_map).to(activations.device) * (1-mask[:,i:i+1,:,:]) + input[:,i:i+1,:,:] * mask[:,i:i+1,:,:]
                same_distri_map = np.random.choice(cen_bin, size=centre_numpy[:,i:i+1,:,:].shape, replace=True, p=cen_hist)
                '''
                cen_logit = self.model_arch(centre_data)
                cen_logit = F.softmax(cen_logit, dim=1)
                cen_logit = - torch.log(cen_logit)
                cen_score = torch.sum(one_hot * cen_logit, dim=1, keepdim=False)

                
                sur_logit = self.model_arch(surround_data)
                sur_logit = F.softmax(sur_logit, dim=1)
                sur_logit = - torch.log(sur_logit)
                sur_score = torch.sum(one_hot * sur_logit, dim=1, keepdim=False)
                alpha = 0.75
                slope[:, i:i+1] = alpha * (score-cen_score) - (1-alpha) * (score-sur_score)
                #slope[:, i:i+1] = cen_score ** alpha / sur_score ** (1-alpha) * score **(1-2*alpha)#(score - sur_score) * cen_score
        # print(slope)
        # slope = slope_norm(slope)
        slope = F.softmax(slope, dim=-1)
        # gap = (1-slope.mean(dim=-1))
        # slope = slope + gap
        # slope = slope ** 2 - gap
        # slope = F.relu(slope)
        # slope = torch.where(slope>0, slope_, torch.zeros_like(slope_))
        '''
        slope_s = torch.zeros_like(slope) #(b, k).to(activations.device)
        slope_ss = torch.zeros_like(slope) #(b, k).to(activations.device)
        sorted_slope, indices_slope = torch.sort(slope, dim=-1, descending=True)
       
        sorted_activations = torch.zeros_like(activations)
        for i in range(sorted_activations.size(0)):
            sorted_activations[i, :, :, :] = activations[i, indices_slope[i, :], :, :]
        m = k
        
        mul_activation = torch.zeros(b, k, 224, 224).to(input.device)
        with torch.no_grad():
            norm_saliency_map_a = torch.zeros_like(input[:, 0:1, :, :])
            mask_a = torch.zeros_like(input[:, 0:1, :, :])
            area = torch.zeros_like(slope_s)
            comm_area = torch.zeros_like(slope_s)
            for i in range(m):
                saliency_map_b = sorted_activations[:, i:i+1, :, :].detach()
                # 归一化
                norm_saliency_map_b = data_norm(saliency_map_b)
                norm_saliency_map_b = F.interpolate(norm_saliency_map_b, size=(224, 224), mode='bilinear', align_corners=False)
                ret_b = norm_saliency_map_b.mean(dim=(-2, -1), keepdim=False)
                mask_b = torch.where(norm_saliency_map_b <= ret_b, 
                                     torch.tensor(0.).to(input.device), 
                                     torch.tensor(1.).to(input.device))
                full_mask = max_data(mask_a, mask_b)
                resi_mask = resi_data(mask_a, mask_b)
                full_area = full_mask.sum(dim=(-2, -1), keepdim=False)
                resi_area = resi_mask.sum(dim=(-2, -1), keepdim=False)
                a_area = mask_a.sum(dim=(-2, -1), keepdim=False)
                b_area = mask_b.sum(dim=(-2, -1), keepdim=False)
                com_mask = common_data(mask_a, mask_b)
                com_area = com_mask.sum(dim=(-2, -1), keepdim=False)
                # area_radio = (a_area + b_area) / full_area
                # area_radio = b_area / full_area

                # res_data = (mask_b * 0.5 + resi_mask * 0.5).repeat(1, 3, 1, 1) * input
                # res_data = ((1-area_radio) * mask_a * 0.5 + area_radio * mask_b * 0.5).repeat(1, 3, 1, 1) * input
                res_data = full_mask.repeat(1, 3, 1, 1) * input
                self.model_arch.zero_grad()
                output = self.model_arch(res_data)# * area_radio
                output = F.softmax(output, dim=-1)
                # output = - torch.log(output)
                score_res = torch.sum(one_hot * output, dim=1, keepdim=False)

                area[:, i:i+1] = full_area
                comm_area[:, i:i+1] = com_area
                slope_s[:, i:i+1] = score_res # (score - score_res)
                # radio[:, i:i+1] = area_radio
                # if i == 0:
                #     slope_s[:, i:i+1] = score - score_res
                #     store = slope_s[:, i:i+1]
                # else:
                #     slope_s[:, i:i+1] = score - score_res - store
                #     store = score - score_res
                
                mul_activation[:, i:i+1, :, :] = data_norm(full_area * (1-score_res) * norm_saliency_map_a + full_area * score_res * norm_saliency_map_b) # resi_area * norm_saliency_map_b
                norm_saliency_map_a = data_norm(full_area * (1-score_res) * norm_saliency_map_a + full_area * score_res * norm_saliency_map_b) #data_norm(norm_saliency_map_a * mask_a + norm_saliency_map_b * mask_b)
                mask_a = max_data(mask_a, mask_b)

        slope_ss[:, 0:1]  = slope_s[:, 0:1]  / (area[:, 0:1]+1)
        for i in range(1, m):
            slope_ss[:, i:i+1] = (slope_s[:, i:i+1] - slope_s[:, i-1:i]*(comm_area[:, i:i+1]+1)/(area[:, i-1:i]+1)) \
               / (area[:, i:i+1] - area[:, i-1:i] + 1) #* area[:, i:i+1]
        
        # print(score)
        # print(sorted_slope)
        slope_ss = F.softmax(slope_ss, dim=-1)# * radio
        # slope_s = slope_norm(slope_s)
        # slope_ss = torch.where(slope_ss <= 0, torch.zeros_like(slope_ss), slope_sss)
        # print(slope_ss)
        # ind = slope.argmax(-1, keepdim=False)
        # saliency_map = mul_activation[:, ind:ind+1, :, :]
        '''
        saliency_map = (slope.view(b, k, 1, 1) * activations).sum(1, keepdim=True)
        # 归一化
        score_saliency_map = data_norm(F.relu(saliency_map))
        score_saliency_map = F.interpolate(score_saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
        return score_saliency_map.detach(), logit.detach()