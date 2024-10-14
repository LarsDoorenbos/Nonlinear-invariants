
from dataclasses import dataclass
import logging
import importlib
import os
from typing import Optional, Tuple, List, Dict, Union, Any, cast

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

# Torch imports
from torch import nn
from torch import Tensor
import torch
from torch.utils.data import DataLoader

# Ignite imports
from ignite.engine import Engine, Events
import ignite.distributed as idist
from ignite.utils import setup_logger
from ignite.metrics import Frequency
from ignite.contrib.metrics import ROC_AUC
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.contrib.handlers import WandBLogger
from ignite.contrib.metrics import GpuInfo

# Local imports
from .utils import archive_code, expanduservars, knn_score
from .model import build_model, VolumePreservingNet
from .dataset_utils import NumpyDataset
from .optimizer import build_optimizer

LOGGER = logging.getLogger(__name__)
Model = Union[VolumePreservingNet, nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]


def _flatten(m: Model) -> VolumePreservingNet:
    if isinstance(m, VolumePreservingNet):
        return m
    elif isinstance(m, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
        return cast(VolumePreservingNet, m.module)
    else:
        raise TypeError("type(m) should be one of (VolumePreservingNet, DataParallel, DistributedDataParallel)")


def find_number_of_invariants(data, pca_variance_percentage):
    pca = PCA()
    pca_variance_ratio = 1 - (pca_variance_percentage / 100)
    _ = pca.fit_transform(data)
    number_of_invariants = np.where(np.cumsum(pca.explained_variance_ratio_) > pca_variance_ratio)[0][0]
    
    number_of_invariants = data.shape[1] - number_of_invariants
    LOGGER.info("%d invariants", number_of_invariants)

    return number_of_invariants, pca.explained_variance_


@torch.no_grad()
def get_train_invariants(network: VolumePreservingNet, train_loader: DataLoader, num_invariants: int):
    network.eval()
    pred_y = []
    for batch in train_loader:
        x = batch[0].to(idist.device())
        pred = network(x).clone().detach()
        pred_y.append(pred)

    pred_y = torch.cat(pred_y).cpu()
    mses = torch.abs(pred_y[:, -num_invariants:] ** 2)
    inv_mses = torch.mean(mses, dim=0)

    return inv_mses


@torch.no_grad()
def get_testset_scores(network: VolumePreservingNet, dataloader: DataLoader, num_invariants: int):
    network.eval()
    inv_errors = []
    for batch in dataloader:
        x = batch[0].to(idist.device())
        pred = network(x).clone().detach()
        inv_errors.append(torch.abs(pred[:, -num_invariants:] ** 2))

    inv_errors = torch.cat(inv_errors, dim=0).cpu()

    return inv_errors


@dataclass
class Trainer:
    
    model: VolumePreservingNet
    optimizer: torch.optim.Optimizer
    lr_scheduler: Union[torch.optim.lr_scheduler.LambdaLR, None]
    invariants: int
    invariant_mses: List

    @property
    def flat_model(self):
        """View of the model without DataParallel wrappers."""
        return _flatten(self.model)

    def train_step(self, engine: Engine, batch) -> dict:

        x, y = batch

        self.model.train()
        
        device = idist.device()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        batch_size = x.shape[0]

        pred_y = self.model(x)

        inv_loss = torch.mean(torch.abs(pred_y[:, -self.invariants:]**2))

        self.optimizer.zero_grad()
        inv_loss.backward()
        self.optimizer.step()

        inv_loss = inv_loss.item()

        pred_y = pred_y.detach().clone()
        pred_y[:, -self.invariants:] = 0.0

        pred_x = self.flat_model.reverse(pred_y)
        rec_loss = torch.nn.functional.mse_loss(pred_x, x)

        self.optimizer.zero_grad()
        rec_loss.backward()
        self.optimizer.step()

        rec_loss = rec_loss.item()
        
        if self.lr_scheduler is not None:
            lr = self.lr_scheduler.get_last_lr()[0]
            self.lr_scheduler.step()
        else:
            lr = self.optimizer.defaults['lr']

        return {"num_items": batch_size, "lr": lr, "inv_loss": inv_loss, "rec_loss": rec_loss}

    @torch.no_grad()
    def test_step(self, _: Engine, batch: Tensor) -> Dict[str, Any]:

        x, y = batch
        x = x.to(idist.device())
        y = y.to(idist.device())

        self.model.eval()
        pred_y = self.model(x)

        pred_y = pred_y.detach().clone()
        pred_y[:, -self.invariants:] = 0.0
        pred_x = self.flat_model.reverse(pred_y)

        return {'y': x, 'y_pred': pred_x}

    @torch.no_grad()
    def auc_step_invs(self, _: Engine, batch: Tensor) -> Dict[str, Any]:

        x, y = batch
        x = x.to(idist.device())
        y = y.to(idist.device())

        self.model.eval()

        pred_y = self.model(x)
        pred_y = pred_y.detach().clone()

        inv_errors = torch.abs(pred_y[:, -self.invariants:]**2)
        inv_errors = inv_errors.sum(dim=1)

        return {'y': y, 'y_pred': inv_errors}

    def objects_to_save(self, engine: Optional[Engine] = None) -> Dict[str, Any]:
        to_save: Dict[str, Any] = {
            "model": self.flat_model,
            "optimizer": self.optimizer,
        }

        if engine is not None:
            to_save["engine"] = engine

        return to_save


def build_engine(trainer: Trainer, output_path: str, train_loader: DataLoader, test_loader: DataLoader, num_invariants: int, explained_variance: List, layer: int, params: dict) -> Engine:
    engine = Engine(trainer.train_step)
    frequency_metric = Frequency(output_transform=lambda x: x["num_items"])
    frequency_metric.attach(engine, "imgs/s", Events.ITERATION_COMPLETED)
    GpuInfo().attach(engine, "gpu")

    engine.state.current_num_invariants = num_invariants

    auc_engine_invs = Engine(trainer.auc_step_invs)
    ROC_AUC().attach(auc_engine_invs, 'auc_invs')

    if idist.get_local_rank() == 0:
        if params["use_logger"]:
            wandb_logger = WandBLogger(project='nonlinear-invariants', name = str(params["class_label"]) + '-' + params["dataset_file"][9:] + '-l' + str(layer))

            wandb_logger.attach_output_handler(
                engine,
                event_name=Events.ITERATION_COMPLETED(every=10),
                tag="training",
                output_transform=lambda x: x,
                global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
            )

            wandb_logger.attach_output_handler(
                auc_engine_invs,
                Events.EPOCH_COMPLETED,
                tag="testing",
                metric_names=["auc_invs"],
                global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
            )

        else: 
            wandb_logger = None

        checkpoint_handler = ModelCheckpoint(
            output_path,
            "model",
            n_saved=1,
            require_empty=False,
            score_function=None,
            score_name=None
        )

    # Display some info every 100 iterations
    @engine.on(Events.ITERATION_COMPLETED(every=100))
    @idist.one_rank_only(rank=0, with_barrier=True)
    def log_info(engine: Engine):
        LOGGER.info(
            "epoch=%d, iter=%d, speed=%.2fimg/s, rec_loss=%.4g, inv_loss=%.4g, gpu:0 util=%.2f%%",
            engine.state.epoch,
            engine.state.iteration,
            engine.state.metrics["imgs/s"],
            engine.state.output["rec_loss"],
            engine.state.output["inv_loss"],
            engine.state.metrics["gpu:0 util(%)"]
        )

    # Save model every 1000 iterations
    @engine.on(Events.ITERATION_COMPLETED(every=10))
    @idist.one_rank_only(rank=0, with_barrier=True)
    def save_model(engine: Engine):
        checkpoint_handler(engine, trainer.objects_to_save(engine))

    # Compute the mse/auc score
    @engine.on(Events.EPOCH_COMPLETED(every=params["max_epochs"]))
    def test(_: Engine):
        LOGGER.info("MSE&AUC computation...")

        trainer.invariant_mses = get_train_invariants(trainer.model, train_loader, engine.state.current_num_invariants)

        auc_engine_invs.run(test_loader, max_epochs=1)

        engine.state.min_mse_test_scores = get_testset_scores(trainer.model, test_loader, engine.state.current_num_invariants)
        engine.state.min_mse_train_scores = torch.clone(trainer.invariant_mses)
                    
        LOGGER.info("AUC: %.4g", auc_engine_invs.state.metrics["auc_invs"])

    return engine


def load(filename: str, trainer: Trainer, engine: Engine):
    LOGGER.info("Loading state from %s...", filename)
    state = torch.load(filename, map_location=idist.device())
    to_load = trainer.objects_to_save(engine)
    ModelCheckpoint.load_objects(to_load, state)


def _build_model(input_dimensionality: int, params: dict) -> Model:
    model: Model = build_model(
        dim = input_dimensionality,
        num_layers = params["num_layers"],
        channel_mults = params["channel_mults"]
    ).to(idist.device())

    # Wrap the model in DataParallel for parallel processing
    if params["multigpu"]:
        model = nn.DataParallel(model)

    return model


def _get_features(base_output_path, params: dict):
    dataset_file: str = params['dataset_file']
    dataset_module = importlib.import_module(dataset_file)
    train_dataset = dataset_module.training_dataset(params["class_label"], params["preprocessing"], params["architecture"], base_output_path)  # type: ignore
    test_in_dataset, test_out_dataset = dataset_module.test_dataset(params["class_label"], params["preprocessing"], params["architecture"], base_output_path)  # type: ignore

    return train_dataset, test_in_dataset, test_out_dataset


def _build_dataloaders(layer, train_dataset, test_in_dataset, test_out_dataset, params) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    
    train_dataset = NumpyDataset(train_dataset[str(layer)], np.zeros(len(train_dataset[str(layer)])))
    test_dataset = NumpyDataset(np.concatenate((test_in_dataset[str(layer)], test_out_dataset[str(layer)]), axis=0), np.concatenate((np.zeros(len(test_in_dataset[str(layer)])), np.ones(len(test_out_dataset[str(layer)])))))

    LOGGER.info("%d datapoints in dataset '%s'", len(train_dataset), params['dataset_file'])
    LOGGER.info("%d datapoints in test dataset '%s'", len(test_dataset), params['dataset_file'])

    dataset_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=params["mp_loaders"]
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=params["mp_loaders"]
    )

    return dataset_loader, test_loader


def run_train(params: dict):
    setup_logger(name=None, format="\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s", reset=True)

    # Create output folder and archive the current code and the parameters there
    base_output_path = expanduservars(params['output_path'])
    os.makedirs(base_output_path, exist_ok=True)
    archive_code(base_output_path)

    LOGGER.info("%d GPUs available", torch.cuda.device_count())
    LOGGER.info("Using %s for %d epochs", params["architecture"], params["max_epochs"])
    LOGGER.info("bs %d, lr %f, pca_var %f", params["batch_size"], params["optim"]["learning_rate"], params["pca_variance_percentage"])

    train_dataset, test_in_dataset, test_out_dataset = _get_features(base_output_path, params)
    scores = np.zeros(len(test_in_dataset['0']) + len(test_out_dataset['0']))
    
    for layer in range(len(train_dataset)) if params["preprocessing"] != 'normalize_last' else range(len(train_dataset)-1):
        output_path = os.path.join(base_output_path, 'layer' + str(layer))

        # Load the datasets
        train_loader, test_loader = _build_dataloaders(layer, train_dataset, test_in_dataset, test_out_dataset, params)

        # Build the model, optimizer, trainer and training engine
        input_dimensionality = train_loader.dataset[0][0].shape[0]
        LOGGER.info("%d dimensions", input_dimensionality)
        
        model = _build_model(input_dimensionality, params)

        optimizer_staff = build_optimizer(params, model, train_loader)
        optimizer = optimizer_staff['optimizer']
        lr_scheduler = optimizer_staff['lr_scheduler']

        number_of_invariants, explained_variance = find_number_of_invariants(train_loader.dataset.get_data(), params["pca_variance_percentage"])

        trainer = Trainer(model, optimizer, lr_scheduler, number_of_invariants, [])
        engine = build_engine(trainer, output_path, train_loader, test_loader, number_of_invariants, explained_variance, layer, params=params)
        
        # Load a model (if requested in params.yml) to continue training from it
        load_from = params.get('load_from', None)
        if load_from is not None:
            fname = [i for i in os.listdir(os.path.join(load_from, 'layer' + str(layer))) if i.startswith('best')]
            load_from = expanduservars(os.path.join(load_from, 'layer' + str(layer), fname[0]))
            load(load_from, trainer=trainer, engine=engine)

        engine.run(train_loader, max_epochs=params["max_epochs"])

        train_scores = engine.state.min_mse_train_scores.numpy()
        layer_scores = engine.state.min_mse_test_scores.numpy()

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

    groundTruthIn = np.array([1 for i in range(len(test_in_dataset['0']))])
    groundTruthOut = np.array([-1 for i in range(len(test_out_dataset['0']))])

    groundTruth = np.append(groundTruthIn, groundTruthOut)
    auc = roc_auc_score(groundTruth, -1 * scores)

    print('final AUC: ', auc)