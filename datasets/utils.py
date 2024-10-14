
import os

import numpy as np

import torch.nn as nn
import torch
import torchvision

from efficientnet_pytorch import EfficientNet

import ignite.distributed as idist


class ResNet_features(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.conv1 = resnet.conv1

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def get_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)

        result = []
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        for i in [2, 3, 4]:
            result.append(locals()["x" + str(i)].mean(dim=(2,3)))

        return result
    

class EfficientNet_features(EfficientNet):
    def get_features(self, inputs):
        features = []

        x = self._swish(self._bn0(self._conv_stem(inputs)))

        x_prev = x
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if (x_prev.shape[1] != x.shape[1] and idx != 0) and idx > 11:
                features.append(x_prev.mean(dim=(2,3)))
            if idx == (len(self._blocks) - 1) and idx > 11:
                features.append(x.mean(dim=(2,3)))
            x_prev = x

        x = self._swish(self._bn1(self._conv_head(x)))
        features.append(x.mean(dim=(2,3)))

        return features

class Encoder_features(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.pos_embedding = encoder.pos_embedding
        self.ln = encoder.ln
        self.dropout = encoder.dropout
        self.layers = encoder.layers
    
    def get_features(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        input = self.dropout(input)
        
        result = []
        for idx, layer in enumerate(self.layers):
            input = layer(input)
            
            if idx in [7, 9, 11]:
                result.append(input.mean(dim=1))
            
        return result
   

class VIT_features(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self._process_input = vit._process_input
        self.class_token = vit.class_token
        self.encoder = Encoder_features(vit.encoder)
        self.heads = vit.heads

    def get_features(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder.get_features(x)

        return x
    

class ConvNext_features(nn.Module):
    def __init__(self, convnext):
        super().__init__()
        self.features = convnext.features
        self.avgpool = convnext.avgpool
        self.classifier = convnext.classifier

    def get_features(self, x):
        result = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in [1, 3, 5, 7]:
                result.append(x.mean(dim=(2,3)))
        
        return result


class Model(torch.nn.Module):
    def __init__(self, architecture):
        super().__init__()
        if 'resnet' in architecture:
            self.backbone = torch.hub.load('pytorch/vision:v0.6.0', architecture, pretrained=True)
            self.backbone.fc = torch.nn.Identity()
        elif 'vitb16' in architecture:
            self.backbone = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
        elif 'convnext-b' in architecture:
            self.backbone = torchvision.models.convnext_base(weights=torchvision.models.ConvNeXt_Base_Weights.DEFAULT)
        elif 'efficientnet' in architecture:
            self.backbone = EfficientNet_features.from_pretrained('efficientnet-b0')
        

class FeatureTrainingDataset(nn.Module):
    def __init__(self, dataset, architecture, preprocessing, output_path):
        super().__init__()
        self.dataset = dataset

        model = Model(architecture)

        if 'resnet' in architecture:
            self.model = ResNet_features(model.backbone.to(idist.device()))
        elif 'vitb16' in architecture:
            self.model = VIT_features(model.backbone.to(idist.device()))
        elif 'convnext' in architecture:
            self.model = ConvNext_features(model.backbone.to(idist.device()))
        elif 'efficientnet' in architecture:
            self.model = model.backbone.to(idist.device())

        num_of_parameters = sum(map(torch.numel, model.parameters()))
        print(f"The {architecture} has {num_of_parameters} trainable parameters.")

        self.preprocessing = preprocessing
        self.output_path = output_path

    def get_training_set(self):
        dataset = self.dataset

        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)
        features = get_latent_vectors(loader, self.model)
        
        if self.preprocessing == 'normalize_last':
            old_features = features[str(len(features) - 1)].copy()
            features[str(len(features) - 1)] = features[str(len(features) - 1)] / np.linalg.norm(features[str(len(features) - 1)], axis=1)[:, None]
            features[str(len(features))] = old_features
            
            features = center_data(features, True, self.output_path)
            
        return features


class FeatureTestDataset(nn.Module):
    def __init__(self, in_dataset, out_dataset, architecture, preprocessing, output_path):
        super().__init__()
        self.in_dataset = in_dataset
        self.out_dataset = out_dataset
        
        model = Model(architecture)

        if 'resnet' in architecture:
            self.model = ResNet_features(model.backbone.to(idist.device()))
        elif 'vitb16' in architecture:
            self.model = VIT_features(model.backbone.to(idist.device()))
        elif 'convnext' in architecture:
            self.model = ConvNext_features(model.backbone.to(idist.device()))
        elif 'efficientnet' in architecture:
            self.model = model.backbone.to(idist.device())

        self.preprocessing = preprocessing
        self.output_path = output_path

    def get_test_set(self):
        in_loader = torch.utils.data.DataLoader(self.in_dataset, batch_size=32, shuffle=False, num_workers=8)
        features_in = get_latent_vectors(in_loader, self.model)

        out_loader = torch.utils.data.DataLoader(self.out_dataset, batch_size=32, shuffle=False, num_workers=8)
        features_out = get_latent_vectors(out_loader, self.model)

        if self.preprocessing == 'normalize_last':
            old_features_in = features_in[str(len(features_in) - 1)].copy()
            features_in[str(len(features_in) - 1)] = features_in[str(len(features_in) - 1)] / np.linalg.norm(features_in[str(len(features_in) - 1)], axis=1)[:, None]
            features_in[str(len(features_in))] = old_features_in
            
            features_in = center_data(features_in, False, self.output_path)

            old_features_out = features_out[str(len(features_out) - 1)].copy()
            features_out[str(len(features_out) - 1)] = features_out[str(len(features_out) - 1)] / np.linalg.norm(features_out[str(len(features_out) - 1)], axis=1)[:, None]
            features_out[str(len(features_out))] = old_features_out
            
            features_out = center_data(features_out, False, self.output_path)

        return features_in, features_out


@torch.no_grad()
def get_latent_vectors(dataloader, model):
    model.eval()
    latent_vectors = {}
    
    for cnt, x in enumerate(dataloader):
        x = x[0].to(idist.device())
        features = model.get_features(x)
        
        for i in range(len(features)):
            if cnt == 0:
                latent_vectors[str(i)] = []    

            latent_vectors[str(i)].append(features[i])

    for i in range(len(features)):
        latent_vectors[str(i)] = torch.cat(latent_vectors[str(i)]).cpu().numpy()

    return latent_vectors        


def center_data(latent_vectors, save, base_output_path):

    for i in range(len(latent_vectors)):
        if save == True:
            output_path = os.path.join(base_output_path, 'layer' + str(i))
            os.makedirs(output_path, exist_ok=True)
            np.save(os.path.join(output_path, 'mean.npy'), np.mean(latent_vectors[str(i)], axis=0))

            latent_vectors[str(i)] = latent_vectors[str(i)] - np.mean(latent_vectors[str(i)], axis=0)
        else:
            mean = np.load(os.path.join(base_output_path, 'layer' + str(i), 'mean.npy'))
            latent_vectors[str(i)] = latent_vectors[str(i)] - mean
    
    return latent_vectors