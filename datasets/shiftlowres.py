
import torch
import torchvision
from torchvision import transforms

from .utils import FeatureTrainingDataset, FeatureTestDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def training_dataset(class_label, preprocessing, architecture, base_output_path):
    dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    features = FeatureTrainingDataset(dataset, architecture, preprocessing, base_output_path).get_training_set()

    return features


def test_dataset(class_label, preprocessing, architecture, base_output_path):
    testSetIn = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    testSetIn, _ = torch.utils.data.random_split(testSetIn, [10000, len(testSetIn)-10000], generator=torch.Generator().manual_seed(1))

    testSetOut = torchvision.datasets.SVHN(root='../data', split='test', download=True, transform=transform)
    testSetOut, _ = torch.utils.data.random_split(testSetOut, [10000, len(testSetOut)-10000], generator=torch.Generator().manual_seed(1))

    return FeatureTestDataset(testSetIn, testSetOut, architecture, preprocessing, base_output_path).get_test_set()


    