import os
import sys
from typing import Dict, Any
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from model_training.utils import load_hydra_config, create_logger
from model_training.train.trainer import DAD3DTrainer
from model_training.model import load_model
from model_training.train.flame_lightning_model import FlameLightningModel
from model_training.data import FlameDataset
from pytorch_lightning import Trainer

logger = create_logger(__name__)


def test(config):
    """
    Test/evaluate a trained model on validation or test dataset.
    
    Args:
        config: Configuration dictionary with model, dataset, and checkpoint paths
    """
    # Create datasets
    # For testing, we typically use val dataset, but can also use test if available
    if "test" in config and config["test"].get("ann_path"):
        logger.info("Using test dataset")
        test_dataset = FlameDataset.from_config(config=config["test"])
        dataset_mode = "test"
    else:
        logger.info("Using validation dataset")
        test_dataset = FlameDataset.from_config(config=config["val"])
        dataset_mode = "val"
    
    # Create a dummy train dataset (required by FlameLightningModel but not used for testing)
    train_dataset = FlameDataset.from_config(config=config["train"])
    
    # Load checkpoint if specified
    checkpoint_path = config.get("checkpoint_path") or config.get("weights_path")
    
    # Load model
    model = load_model(config["model"], config["constants"])
    
    # Create Lightning model
    # If checkpoint is specified, set it in config so _load_model can use it
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        config["load_weights"] = True
        config["weights_path"] = checkpoint_path
    else:
        logger.warning("No checkpoint path specified. Testing with untrained model weights.")
    dad3d_net = FlameLightningModel(model=model, config=config, train=train_dataset, val=test_dataset)
    
    # Create trainer (for testing, we don't need all training callbacks)
    from model_training.train.utils import create_trainer
    config["wandb"]["enable"] = False
    trainer = create_trainer(config)
    
    # Run validation/test
    logger.info(f"Running evaluation on {dataset_mode} dataset...")
    trainer.validate(dad3d_net)
    
    # If test dataset is available, also run test
    if dataset_mode == "test":
        logger.info("Running test evaluation...")
        trainer.test(dad3d_net)
    
    logger.info("Evaluation completed")


def prepare_experiment(hydra_config: DictConfig) -> Dict[str, Any]:
    """
    Prepare experiment configuration from Hydra config.
    Similar to train.py but adapted for testing.
    """
    experiment_dir = os.getcwd()
    save_path = os.path.join(experiment_dir, "test_config.yaml")
    OmegaConf.set_struct(hydra_config, False)
    
    # If checkpoint_path is not in config, try to get it from experiment folder
    if "checkpoint_path" not in hydra_config and "experiment" in hydra_config:
        exp_folder = hydra_config["experiment"].get("folder", "")
        exp_name = hydra_config["experiment"].get("name", "")
        if exp_folder and exp_name:
            # Try to find the best checkpoint
            checkpoint_dir = os.path.join(exp_folder, exp_name, "checkpoints")
            if os.path.exists(checkpoint_dir):
                # Look for last checkpoint or best checkpoint
                checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
                if checkpoint_files:
                    # Prefer last checkpoint, or best if available
                    last_ckpt = [f for f in checkpoint_files if "last" in f.lower()]
                    if last_ckpt:
                        hydra_config["checkpoint_path"] = os.path.join(checkpoint_dir, last_ckpt[0])
                    else:
                        # Use the most recent checkpoint
                        checkpoint_paths = [os.path.join(checkpoint_dir, f) for f in checkpoint_files]
                        checkpoint_paths.sort(key=os.path.getmtime, reverse=True)
                        hydra_config["checkpoint_path"] = checkpoint_paths[0]
                    logger.info(f"Auto-detected checkpoint: {hydra_config['checkpoint_path']}")
    
    if "experiment" in hydra_config:
        hydra_config["experiment"]["folder"] = experiment_dir
    
    logger.info(OmegaConf.to_yaml(hydra_config, resolve=True))
    config = load_hydra_config(hydra_config)
    
    with open(save_path, "w") as f:
        OmegaConf.save(config=config, f=f.name)
    
    return config


def parse_checkpoint_path_from_argv():
    """
    Parse checkpoint_path from sys.argv before Hydra processes it.
    Handles both --checkpoint_path and checkpoint_path= formats.
    """
    checkpoint_path = None
    args_to_remove = []
    
    # Check for --checkpoint_path argument
    for i, arg in enumerate(sys.argv):
        if arg == "--checkpoint_path" and i + 1 < len(sys.argv):
            checkpoint_path = sys.argv[i + 1]
            # Mark for removal
            args_to_remove.append(i)
            args_to_remove.append(i + 1)
            break
        elif arg.startswith("--checkpoint_path="):
            checkpoint_path = arg.split("=", 1)[1]
            args_to_remove.append(i)
            break
        elif arg.startswith("checkpoint_path="):
            checkpoint_path = arg.split("=", 1)[1]
            args_to_remove.append(i)
            break
    
    # Remove arguments in reverse order to maintain indices
    for i in sorted(args_to_remove, reverse=True):
        sys.argv.pop(i)
    
    return checkpoint_path


@hydra.main(config_name="train", config_path="model_training/config")
def run_test(hydra_config: DictConfig) -> None:
    """
    Main entry point for testing.
    Uses Hydra to load configuration, similar to train.py.
    """
    config = prepare_experiment(hydra_config)
    
    # Check for checkpoint_path from command line (stored in environment)
    cmd_checkpoint_path = os.environ.get("TEST_CHECKPOINT_PATH")
    if cmd_checkpoint_path:
        config["checkpoint_path"] = cmd_checkpoint_path
        logger.info(f"Using checkpoint path from command line: {cmd_checkpoint_path}")
        # Clean up environment variable
        del os.environ["TEST_CHECKPOINT_PATH"]
    
    if "experiment" in config:
        logger.info("Experiment dir %s" % config["experiment"]["folder"])
    
    # Validate checkpoint path
    checkpoint_path = config.get("checkpoint_path") or config.get("weights_path")
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            logger.error("Please specify a valid checkpoint_path:")
            logger.error("  python test.py --checkpoint_path path/to/checkpoint.ckpt")
            logger.error("  or")
            logger.error("  python test.py checkpoint_path=path/to/checkpoint.ckpt")
            return
    else:
        logger.warning("No checkpoint_path specified. Will test with untrained weights.")
        logger.warning("To specify a checkpoint, use:")
        logger.warning("  python test.py --checkpoint_path path/to/checkpoint.ckpt")
        logger.warning("  or")
        logger.warning("  python test.py checkpoint_path=path/to/checkpoint.ckpt")
    
    print(config)
    test(config)


if __name__ == "__main__":
    print("Running test/evaluation")
    # Parse checkpoint_path from command line before Hydra processes arguments
    cmd_checkpoint_path = parse_checkpoint_path_from_argv()
    
    # Store it in a way that run_test can access it
    # We'll use a global or pass it through environment
    if cmd_checkpoint_path:
        os.environ["TEST_CHECKPOINT_PATH"] = cmd_checkpoint_path
    
    run_test()

# CUDA_VISIBLE_DEVICES=2,3 python test.py --checkpoint_path experiments/train/2025-12-21-14-35-05/train_on_all_fusion_on_all/checkpoints/epoch_0106-valid_metrics_reproject_nme_2d_1.6748.ckpt
