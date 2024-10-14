
import torch
import torchvision
from torchvision import transforms

from .utils import FeatureTrainingDataset, FeatureTestDataset

mvteclist = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def training_dataset(class_label, preprocessing, architecture, base_output_path):
    dataset = torchvision.datasets.ImageFolder(root="../data/mvtec/" + mvteclist[class_label] + "/train/good/", transform=transform)
    features = FeatureTrainingDataset(dataset, architecture, preprocessing, base_output_path).get_training_set()

    return features


def test_dataset(class_label, preprocessing, architecture, base_output_path):
    in_dataset = torchvision.datasets.ImageFolder(root="../data/mvtec/" + mvteclist[class_label] + "/test/good/", transform=transform)
    out_dataset = torchvision.datasets.ImageFolder(root="../data/mvtec/" + mvteclist[class_label] + "/test/out/", transform=transform)
    
    return FeatureTestDataset(in_dataset, out_dataset, architecture, preprocessing, base_output_path).get_test_set()
