
import logging
import importlib
import os
from typing import Union
import time

import numpy as np
from sklearn.metrics import roc_auc_score

# Torch imports
from torch import nn
import torch
from torch.optim.lr_scheduler import MultiplicativeLR

# Ignite imports
from ignite.utils import setup_logger

# Local imports
from nl_invariants.trainer import Trainer, knn_score, get_train_invariants, get_testset_scores, find_number_of_invariants, _build_model, load, _build_dataloaders
from nl_invariants.utils import archive_code, expanduservars
from nl_invariants.model import VolumePreservingNet

LOGGER = logging.getLogger(__name__)
Model = Union[VolumePreservingNet, nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]


def unnormalize(image):
    image[0] = image[0] * 0.229 + 0.485
    image[1] = image[1] * 0.224 + 0.456
    image[2] = image[2] * 0.225 + 0.406
    return image


def _get_features(params: dict):
    base_output_path = params["load_from"]
    dataset_file: str = params['dataset_file']
    dataset_module = importlib.import_module(dataset_file)
    train_dataset = dataset_module.training_dataset(params["class_label"], params["preprocessing"], params["architecture"], base_output_path)  # type: ignore
    test_in_dataset, test_out_dataset = dataset_module.test_dataset(params["class_label"], params["preprocessing"], params["architecture"], base_output_path)  # type: ignore

    return train_dataset, test_in_dataset, test_out_dataset


def run_ms_eval(local_rank: int, params: dict):
    setup_logger(name=None, format="\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s", reset=True)

    # Create output folder and archive the current code and the parameters there
    base_output_path = expanduservars(params['output_path'])
    os.makedirs(base_output_path, exist_ok=True)
    archive_code(base_output_path)

    LOGGER.info("%d GPUs available", torch.cuda.device_count())
    LOGGER.info("Using %s for %d epochs", params["architecture"], params["max_epochs"])

    train_dataset, test_in_dataset, test_out_dataset = _get_features(params)

    groundTruthIn = np.array([1 for i in range(len(test_in_dataset['0']))])
    groundTruthOut = np.array([-1 for i in range(len(test_out_dataset['0']))])

    groundTruth = np.append(groundTruthIn, groundTruthOut)
    
    scores = np.zeros(len(test_in_dataset['0']) + len(test_out_dataset['0']))

    for layer in range(len(train_dataset)) if params["preprocessing"] != 'normalize_last' else range(len(train_dataset)-1):
        output_path = os.path.join(base_output_path, 'layer' + str(layer))
        os.makedirs(output_path, exist_ok=True)
    
        start = time.time()
        # Load the datasets
        train_loader, test_loader = _build_dataloaders(layer, train_dataset, test_in_dataset, test_out_dataset, params)

        # Build the model, optimizer, trainer and training engine
        input_dimensionality = train_loader.dataset[0][0].shape[0]
        LOGGER.info("%d dimensions", input_dimensionality)
        
        model = _build_model(input_dimensionality, params)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.9)

        number_of_invariants, explained_variance = find_number_of_invariants(train_loader.dataset.get_data(), params["pca_variance_percentage"])

        trainer = Trainer(model, optimizer, scheduler, number_of_invariants, [])

        # Finds model checkpoints in the layer folders. TODO: handle multiple checkpoints
        load_from = params.get('load_from', None)
        if load_from is not None:   
            fname = [i for i in os.listdir(os.path.join(load_from, 'layer' + str(layer))) if i.startswith('model')]
            print(os.path.join(load_from, 'layer' + str(layer), fname[0]))
            load_from = expanduservars(os.path.join(load_from, 'layer' + str(layer), fname[0]))
            load(load_from, trainer=trainer, engine=None)

        train_scores = get_train_invariants(trainer.model, train_loader, number_of_invariants).numpy()
        layer_scores = get_testset_scores(trainer.model, test_loader, number_of_invariants).numpy()

        # kNN score
        if params["preprocessing"] == 'normalize_last' and layer == len(train_dataset) - 2:
            train_knn_scores = knn_score(train_dataset[str(layer + 1)], train_dataset[str(layer + 1)], params["k"], train=True)
            in_scores = knn_score(train_dataset[str(layer + 1)], test_in_dataset[str(layer + 1)], params["k"])
            out_scores = knn_score(train_dataset[str(layer + 1)], test_out_dataset[str(layer + 1)].copy(order='C'), params["k"])
        else:
            train_knn_scores = knn_score(train_dataset[str(layer)], train_dataset[str(layer)], params["k"], train=True)
            in_scores = knn_score(train_dataset[str(layer)], test_in_dataset[str(layer)], params["k"])
            out_scores = knn_score(train_dataset[str(layer)], test_out_dataset[str(layer)].copy(order='C'), params["k"])

        knn_scores = np.concatenate((in_scores, out_scores)) / np.mean(train_knn_scores) * len(train_scores)

        scores += (knn_scores + np.sum(layer_scores / train_scores, axis=1))

    auc = roc_auc_score(groundTruth, -1 * scores)

    print('final AUC: ', auc)