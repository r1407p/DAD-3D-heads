import pickle
import os
from model_training.train.utils import any2device
from tqdm import tqdm
from collections import defaultdict
import torch
import hydra
from omegaconf import DictConfig
from model_training.utils import load_hydra_config, create_logger
from model_training.train.flame_lightning_model import FlameLightningModel
from model_training.data import FlameDataset

import os
from typing import Dict, Any
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from model_training.utils import load_hydra_config, create_logger
from model_training.train.trainer import DAD3DTrainer
from model_training.model import load_model
from model_training.train.flame_lightning_model import FlameLightningModel
from model_training.data import FlameDataset
from collections import defaultdict
import numpy as np
from model_training.model.utils import unravel_index, normalize_to_cube, load_from_lighting
from model_training.train.loss_module import LossModule


def analyze(config):
    base_dir = "output_results/baseline"
    files= os.listdir(base_dir)
    baseline_data = []
    for file in tqdm(files):
        with open(os.path.join(base_dir, file), "rb") as f:
            data = pickle.load(f)
        baseline_data.append(any2device(data, "cpu"))

    base_dir = "output_results/no_fusion"
    files= os.listdir(base_dir)
    no_fusion_data = []
    for file in tqdm(files):
        with open(os.path.join(base_dir, file), "rb") as f:
            data = pickle.load(f)
        no_fusion_data.append(any2device(data, "cpu"))

    # tidy up data
    baseline_target_data = defaultdict(list)
    baseline_output_data = defaultdict(list)
    for baseline_d in baseline_data:
        for key, value in baseline_d["target_dict"].items():
            baseline_target_data[key].append(value)
        for key, value in baseline_d["output_dict"].items():
            baseline_output_data[key].append(value)

    for key, value in baseline_target_data.items():
        baseline_target_data[key] = torch.cat(value, dim=0)
    for key, value in baseline_output_data.items():
        baseline_output_data[key] = torch.cat(value, dim=0)

    no_fusion_target_data = defaultdict(list)
    no_fusion_output_data = defaultdict(list)
    for no_fusion_d in tqdm(no_fusion_data):
        for key, value in no_fusion_d["target_dict"].items():
            no_fusion_target_data[key].append(value)
        for key, value in no_fusion_d["output_dict"].items():
            no_fusion_output_data[key].append(value)
    for key, value in no_fusion_target_data.items():
        no_fusion_target_data[key] = torch.cat(value, dim=0)
    for key, value in no_fusion_output_data.items():
        no_fusion_output_data[key] = torch.cat(value, dim=0)

    # train_dataset = FlameDataset.from_config(config=config["train"])
    # val_dataset = FlameDataset.from_config(config=config["val"])
    # model = load_model(config["model"], config["constants"])
    # dad3d_net = FlameLightningModel(model=model, config=config, train=train_dataset, val=val_dataset)
    from model_training.metrics.keypoints import keypoints_nme
    nme_3d_ori = keypoints_nme(baseline_output_data["pred_3d_vertices"], baseline_target_data["TARGET_3D_MODEL_VERTICES"], reduce=None)
    nme_3d_no_fusion = keypoints_nme(no_fusion_output_data["pred_3d_vertices"], no_fusion_target_data["TARGET_3D_MODEL_VERTICES"], reduce=None)

    
    flame_indices = {}
    for key, value in config["train"]["flame_indices"]["files"].items():
        flame_indices[key] = np.load(os.path.join(config["train"]["flame_indices"]["folder"], value))
    tmpa = no_fusion_output_data["pred_3d_vertices"][:, flame_indices["face"]]
    tmpb = no_fusion_target_data["TARGET_3D_MODEL_VERTICES"][:, flame_indices["face"]]
    nme_3d_no_fusion = keypoints_nme(tmpa, tmpb, reduce=None)
    tmpa = baseline_output_data["pred_3d_vertices"][:, flame_indices["face"]]
    tmpb = baseline_target_data["TARGET_3D_MODEL_VERTICES"][:, flame_indices["face"]]
    nme_3d = keypoints_nme(tmpa, tmpb, reduce=None)

    from model_training.losses import IoULoss
    iou_loss = IoULoss()
    iou_loss_value = iou_loss(no_fusion_output_data["OUTPUT_LANDMARKS_HEATMAP"], no_fusion_target_data["TARGET_LANDMARKS_HEATMAP"])
    iou_loss_value_baseline = iou_loss(baseline_output_data["OUTPUT_LANDMARKS_HEATMAP"], baseline_target_data["TARGET_LANDMARKS_HEATMAP"])
    # calculate cov iou_loss_value_baseline, nme_3d_ori
    # cov = np.cov(iou_loss_value_baseline.mean(axis=1), nme_3d_ori)
    # print(f"original cov: {cov}")
    cov = np.cov(iou_loss_value.mean(axis=1), nme_3d_no_fusion)
    print(f"no fusion cov: {cov}")
    cov = np.cov(iou_loss_value_baseline.mean(axis=1), nme_3d)
    print(f"baseline cov: {cov}")
    loss_module = LossModule.from_config(config["loss"])

    loss_3d_baseline = loss_module.criterions[1](baseline_output_data["OUTPUT_3DMM_PARAMS"], baseline_target_data["TARGET_3D_MODEL_VERTICES"])
    loss_3d_no_fusion = loss_module.criterions[1](no_fusion_output_data["OUTPUT_3DMM_PARAMS"], no_fusion_target_data["TARGET_3D_MODEL_VERTICES"])
    
    # calculate mse between pred_vertices_baseline and gt
    cov = np.cov(loss_3d_no_fusion.mean(axis=1), iou_loss_value.mean(axis=1))
    print(f"loss 3d cov: {cov}")
    cov = np.cov(loss_3d_baseline.mean(axis=1), iou_loss_value_baseline.mean(axis=1))
    print(f"loss 3d cov: {cov}")
    # correlation between loss_3d and iou_loss_value
    corr = np.corrcoef(loss_3d_no_fusion.mean(axis=1), iou_loss_value.mean(axis=1))
    print(f"loss 3d correlation: {corr}")
    corr = np.corrcoef(loss_3d_baseline.mean(axis=1), iou_loss_value_baseline.mean(axis=1))
    print(f"loss 3d correlation: {corr}")

    # axis 0 is the batch dimension
    corr = np.corrcoef(loss_3d_no_fusion[:, 0], iou_loss_value.mean(axis=1))
    print(f"loss 3d cov: {cov}")
    corr = np.corrcoef(loss_3d_baseline[:, 0], iou_loss_value_baseline.mean(axis=1))
    print(f"loss 3d cov: {cov}")
    # axis 1 is the batch dimension
    
    print(f"loss 3d correlation: {corr}")
    corr = np.corrcoef(loss_3d_baseline[0], iou_loss_value_baseline[0])
    print(f"loss 3d correlation: {corr}")



from model_training.metrics.keypoints import FailureRate, KeypointsNME


def prepare_experiment(hydra_config: DictConfig) -> Dict[str, Any]:
    experiment_dir = os.getcwd()
    save_path = os.path.join(experiment_dir, "experiment_config.yaml")
    OmegaConf.set_struct(hydra_config, False)
    hydra_config["yaml_path"] = save_path
    hydra_config["experiment"]["folder"] = experiment_dir
    print(OmegaConf.to_yaml(hydra_config, resolve=True))
    config = load_hydra_config(hydra_config)
    with open(save_path, "w") as f:
        OmegaConf.save(config=config, f=f.name)
    return config

config = None
@hydra.main(config_name="train", config_path="model_training/config")
def run_experiment(hydra_config: DictConfig) -> None:
    config = prepare_experiment(hydra_config)
    print(("Experiment dir %s" % config["experiment"]["folder"]))
    analyze(config)

if __name__ == "__main__":
    run_experiment()

# python analyze.py --base_dir "output_results/baseline"