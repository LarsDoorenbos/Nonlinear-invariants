
import torch
import torch.utils
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
    trainSet = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    indList = [class_label]
    train = torch.tensor([1 if trainSet.targets[i] in indList  else 0 for i in range(len(trainSet))])

    dataset = torch.utils.data.Subset(trainSet, train.nonzero())
    features = FeatureTrainingDataset(dataset, architecture, preprocessing, base_output_path).get_training_set()

    return features


def test_dataset(class_label, preprocessing, architecture, base_output_path):
    indList = [class_label]

    testSetIn = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    test = torch.tensor([1 if testSetIn.targets[i] in indList  else 0 for i in range(len(testSetIn))])
    testSetIn = torch.utils.data.Subset(testSetIn, test.nonzero())

    testSetOut = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    test = torch.tensor([1 if testSetOut.targets[i] not in indList  else 0 for i in range(len(testSetOut))])
    testSetOut = torch.utils.data.Subset(testSetOut, test.nonzero())
    testSetOut, _ = torch.utils.data.random_split(testSetOut, [1000, len(testSetOut)-1000], generator=torch.Generator().manual_seed(42))

    return FeatureTestDataset(testSetIn, testSetOut, architecture, preprocessing, base_output_path).get_test_set()


    