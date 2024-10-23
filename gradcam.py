import numpy as np
import cv2
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
def data_norm(data):
    b, k, u, v = data.shape
    data_min = data.view(b, -1).min(dim=-1, keepdim=True).values.view(b, 1, 1, 1)
    data_max = data.view(b, -1).max(dim=-1, keepdim=True).values.view(b, 1, 1, 1)
    denominator = torch.where(
        (data_max-data_min)!=0., data_max-data_min, torch.tensor(1.).to(data.device))

    return (data - data_min) / denominator

class ISGCAM(GradCAM): #index-sort-guide margin cam
    
    def __init__(self, model_dict, verbose=False):
        super(ISGCAM, self).__init__(model_dict, verbose)

    def forward(self, input, class_idx=None, retain_graph=False):
        
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        if class_idx is None:
            class_idx = logit.argmax(dim=1, keepdim=True)

        # logit = F.softmax(logit, dim=1)
        # logit = torch.log(logit)

        one_hot = torch.zeros_like(logit).to(logit.device)
        one_hot = one_hot.scatter_(1, class_idx, 1)
        score = torch.sum(one_hot * logit, dim=-1).view(-1, 1)
        # class_num = logit.size(1)
        # score = logit.sum(1)
        # print("score:",score)

        self.model_arch.zero_grad()
        # score.backward(retain_graph=retain_graph)
        # gradients = self.gradients['value'] # dS/dA
        activations = self.activations['value'] # A
        b, k, u, v = activations.size()
        
        slope_ori = torch.zeros(b, k).to(activations.device)
        slope_contrast = torch.zeros(b, k).to(activations.device)
        self.model_arch.zero_grad()
        with torch.no_grad():
            for i in range(k):
                saliency_map = torch.unsqueeze(activations[:, i, :, :], 1).detach()
                saliency_map = F.interpolate(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
                # norm
                norm_saliency_map = data_norm(saliency_map)
                # threshold = np.percentile(norm_saliency_map.cpu(), 30)
                # ret, th = cv2.threshold((norm_saliency_map[0] * 255).cpu().numpy().transpose(1,2,0).astype('uint8'), 
                #                         0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                ret = norm_saliency_map.mean(dim=(-2, -1), keepdim=False)
                drop_data = torch.where(norm_saliency_map.repeat(1, 3, 1, 1) < (ret), 
                                        torch.zeros_like(input), input)
                # norm_saliency_map = 1 - norm_saliency_map
                # norm_saliency_map[norm_saliency_map < 0.5] = 0 

                sub_logit = self.model_arch(drop_data)
                # sub_logit = F.softmax(sub_logit, dim=1)
                # sub_logit = - torch.log(sub_logit)


                target_score = torch.sum(one_hot * sub_logit, dim=-1, keepdim=False)
                predict_class = sub_logit.argmax(dim=1, keepdim=True)
                pred_one_hot = torch.zeros_like(sub_logit).to(sub_logit.device)
                pred_one_hot = pred_one_hot.scatter_(1, predict_class, 1)
                pred_score = torch.sum(pred_one_hot * sub_logit, dim=-1, keepdim=False)
                slope_ori[:, i:i+1] = target_score
                slope_contrast[:, i:i+1] = torch.exp(target_score - pred_score + 1)

            
            sorted_slope, indices_slope = torch.sort(slope_ori, dim=-1, descending=True)
            sorted_contrast_slope = torch.zeros_like(slope_contrast)
            sorted_activations = torch.zeros_like(activations)

            for i in range(sorted_activations.size(0)):
                sorted_activations[i, :, :, :] = activations[i, indices_slope[i, :], :, :]
                sorted_contrast_slope[i, :] = slope_contrast[i, indices_slope[i, :]]
            accum_activation = sorted_activations[:, 0:1, :, :]
            accm_slope = torch.zeros_like(sorted_contrast_slope)
            for i in range(1, k):
                accum_activation = accum_activation + sorted_activations[:, i:i+1, :, :]
                norm_accum_activation = F.interpolate(accum_activation, size=(224, 224), mode='bilinear', align_corners=False)
                # norm
                norm_accum_activation = data_norm(norm_accum_activation)
                ret = norm_accum_activation.mean(dim=(-2, -1), keepdim=False)
                drop_data = torch.where(norm_accum_activation.repeat(1, 3, 1, 1) < (ret), 
                                        torch.zeros_like(input), input)
                sub_logit = self.model_arch(drop_data)
                predict_class = sub_logit.argmax(dim=1, keepdim=True)
                pred_one_hot = torch.zeros_like(sub_logit).to(sub_logit.device)
                pred_one_hot = pred_one_hot.scatter_(1, predict_class, 1)
                target_score = torch.sum(one_hot * sub_logit, dim=-1, keepdim=False)
                pred_score = torch.sum(pred_one_hot * sub_logit, dim=-1, keepdim=False)
                accm_slope[:, i:i+1] = sorted_contrast_slope[:, i:i+1] / torch.exp(target_score - pred_score + 1)

        # print(slope)
        # slope = F.softmax(slope, dim=-1)
        # slope = F.softmax(slope, dim=-1).view(b, k, 1, 1)
        # slope = slope_norm(slope).view(b, k, 1, 1)
        saliency_map = (accm_slope.view(b, k, 1, 1) * sorted_activations).sum(1, keepdim=True)
        score_saliency_map = F.relu(saliency_map)
        # 
        score_saliency_map = data_norm(score_saliency_map)
        score_saliency_map = F.interpolate(score_saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
        return score_saliency_map.detach(), logit.detach()