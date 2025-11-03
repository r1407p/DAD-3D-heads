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
logger = create_logger(__name__)

torch.autograd.set_detect_anomaly(True)


def train(config):
    train_dataset = FlameDataset.from_config(config=config["train"])
    val_dataset = FlameDataset.from_config(config=config["val"])
    model = load_model(config["model"], config["constants"])
    """
    (Pdb) type(model)
<class 'model_training.model.flame_regression.FlameRegression'>
(Pdb) config["model"]
{
    '_target_': 'model_training.model.flame_regression.FlameRegression', 
    'model_config':{
        'backbone': 'resnet50', 
        'pretrained': True, 
        'num_filters': 256, 
        'num_channels': 3, 
        'num_classes': 68, 
        'img_size': 256, 
        'conv_block': 'regular', 
        'limit_value': 3
    }
}
(Pdb) config["constants"]
{
    'shape': 300, 
    'expression': 100, 
    'jaw': 3, 
    'rotation': 6, 
    'eyeballs': 0, 
    'neck': 0, 
    'translation': 3, 
    'scale': 1
}
"""
    # load weight

    model = torch.jit.load("/home/cytseng/git/DAD-3DHeads/experiments/train/2025-10-22-00-38-57/academic_experiment/checkpoints/epoch_0119-valid_metrics_reproject_nme_2d_1.7283.trcd")
    model = torch.jit.load("/home/cytseng/git/DAD-3DHeads/experiments/train/2025-10-26-18-08-27/predict_but_no_fusion/checkpoints/epoch_0105-valid_metrics_reproject_nme_2d_2.7243.trcd")

    dad3d_net = FlameLightningModel(model=model, config=config, train=train_dataset, val=val_dataset)
    dad3d_trainer = DAD3DTrainer(dad3d_net, config)
    # dad3d_trainer.fit()
    result = dad3d_trainer.trainer.validate(dad3d_net)
    breakpoint()
    dad3d_trainer.dad3d_net = dad3d_trainer.dad3d_net.to("cuda")
    dad3d_trainer.dad3d_net.eval()

    outputs = defaultdict(list)
    targets = defaultdict(list)
    with torch.no_grad():
        for batch in dad3d_trainer.dad3d_net.val_dataloader():
            
            tmp = dad3d_trainer.dad3d_net.validation_step(batch, 0)
            loss = tmp["loss"]
            metrics = tmp["metrics_dict"]
            output = tmp["output_dict"]
            target = tmp["target_dict"]
            for key, value in output.items():
                outputs[key].append(value)
            for key, value in target.items():
                targets[key].append(value)

    breakpoint()

def prepare_experiment(hydra_config: DictConfig) -> Dict[str, Any]:
    experiment_dir = os.getcwd()
    save_path = os.path.join(experiment_dir, "experiment_config.yaml")
    OmegaConf.set_struct(hydra_config, False)
    hydra_config["yaml_path"] = save_path
    hydra_config["experiment"]["folder"] = experiment_dir
    logger.info(OmegaConf.to_yaml(hydra_config, resolve=True))
    config = load_hydra_config(hydra_config)
    with open(save_path, "w") as f:
        OmegaConf.save(config=config, f=f.name)
    return config


@hydra.main(config_name="train", config_path="model_training/config")
def run_experiment(hydra_config: DictConfig) -> None:
    config = prepare_experiment(hydra_config)
    logger.info("Experiment dir %s" % config["experiment"]["folder"])
    train(config)


if __name__ == "__main__":
    run_experiment()
