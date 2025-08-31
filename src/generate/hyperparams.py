from dataclasses import dataclass
from src.utils.save_datapoints import SynWriter
from typing import Literal

@dataclass(frozen=True)
class Generation:

    # Geneation params
    per_class: int = 30000
    batch_gen: int = 512

    # Denoiser specific and ddim params
    num_classes: int = 10
    guidance_scale: float = 4
    eta: float = 0.0
    denoising_steps: int = 5

    # margin params
    margin_eps: float = 0.2
    boundary_scale: float = 2

    # seed
    run_seed: int = 123
    device: str = "cuda"

    # writer
    writer: SynWriter = None

@dataclass(frozen=True)
class train_params:
    manifest_root: str
    steps_per_epoch: int = 50000 // 256
    batch_size: int = 256
    mmin: float = 0.0
    mmax: float = 1.0
    tau_keep: float = 0.0
    epochs : int = 150
    lr: float = 0.001
    alpha_step= 2/255
    epsilon: float = 8/255
    attack_iters: int = 10
    step_size: float = 0.003

@dataclass(frozen=True)
class Evaluate:
    batch_size: int = 256
    norm: Literal["l_inf"] = "l_inf"
    epsilon: float = 8 / 255
    attack_iters: int = 20
    step_size: float = 0.01