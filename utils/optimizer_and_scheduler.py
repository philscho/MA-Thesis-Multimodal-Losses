import math
import importlib
from omegaconf import OmegaConf  # the configs required by functions are based on this

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def get_optimizer_class(optim_name: str) -> Optimizer:
    """
    Get the optmizer instance from pytorch
    """
    if optim_name is not None:
        module = importlib.import_module("torch.optim")
        optim = getattr(module, optim_name)
        return optim


def get_scheduler_class(scheduler_name: str) -> LRScheduler:
    """
    Get the scheduler instance from pytorch
    """
    if scheduler_name is not None:
        module = importlib.import_module("torch.optim.lr_scheduler")
        scheduler = getattr(module, scheduler_name)
        return scheduler


def get_optimizer(optimizer_config: OmegaConf, params) -> Optimizer:
    """
    Get optimizer. The lr parameter is separate to expose it and make life easier
    """
    if optimizer_config.name is not None:
        optim_class = get_optimizer_class(optimizer_config.name)
        return optim_class(
            params=params,
            lr=optimizer_config.lr,
            **optimizer_config.kwargs,
        )


def get_scheduler(scheduler_config: OmegaConf, optim: Optimizer) -> LRScheduler:
    """
    Returns a scheduler based on the config.
    The scheduler config should have the following structure.
    """
    if scheduler_config.name not in ["SequentialLR", "ChainedScheduler"]:
        if scheduler_config.name == "CosineWarmup":
            return get_custom_cosine_schedule_with_warmup(
                    optimizer=optim, **scheduler_config.kwargs)
        else:
            scheduler_class = get_scheduler_class(scheduler_config.name)
            scheduler = scheduler_class(optimizer=optim, **scheduler_config.kwargs)
            return scheduler
    else:
        outerscheduler = get_scheduler_class(scheduler_config.name)
        all_sub_schedulers = [
            get_scheduler(s, optim) for s in scheduler_config.sub_schedulers
        ]
        return outerscheduler(
            optimizer=optim, schedulers=all_sub_schedulers, **scheduler_config.kwargs
        )


def get_custom_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    initial_lr: float,
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return current_step / num_warmup_steps * (1 - initial_lr) + initial_lr
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * 
            (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)))
        )
    
    return LambdaLR(optimizer, lr_lambda)





if __name__ == "__main__":

    config = """
    
    optimizer:
      name: 'AdamW'
      lr: 0.0002
      lr: 1.0
      kwargs: 
        betas: [0.99,0.95]

         
    # scheduler:
    #   name: SequentialLR
    #   kwargs:
    #     milestones: [10]
    #   sub_schedulers:
    #     - name: 'CosineAnnealingLR'
    #       kwargs:
    #         T_max: 5
    #     - name: 'ExponentialLR'
    #       kwargs:
    #         gamma: 0.1

    
    scheduler:
      name: CyclicLR
      kwargs:
        base_lr: 0.0001
        max_lr: 0.0003
        mode: 'triangular2'
    """

    import matplotlib.pyplot as plt

    # scheduler:
    #   name: SequentialLR
    #   kwargs:
    #     milestones: [2]
    #   sub_schedulers:
    #     - name: 'ConstantLR'
    #       kwargs:
    #         factor: 0.1
    #     - name: 'CosineAnnealingLR'
    #       kwargs:
    #         T_max: 5

    config = OmegaConf.create(config)
    print(OmegaConf.to_yaml(config))

    net = torch.nn.Sequential(torch.nn.Linear(100, 50), torch.nn.Linear(50, 100))
    optimizer = get_optimizer(config.optimizer, params=net.parameters())
    scheduler = get_scheduler(config.scheduler, optimizer)

    # optimizer = torch.optim.AdamW(params=net.parameters(),lr=1.0,betas=[0.99,0.95])
    # scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer,total_iters=3)
    dummy_inputs = torch.randn(32, 100)

    print("starting from :", optimizer.state_dict()["param_groups"][0]["lr"])

    lrs = []
    with torch.enable_grad():
        for epoch in range(5000):
            optimizer.zero_grad()
            outs = net(dummy_inputs)
            loss = torch.nn.functional.mse_loss(outs, dummy_inputs)
            loss.backward()
            optimizer.step()
            print(f'Step {epoch} :  {optimizer.state_dict()["param_groups"][0]["lr"]}')
            scheduler.step()
            # print(optimizer.state_dict()["param_groups"][0]["lr"])
            print(f"Step {epoch} :  {scheduler.get_last_lr()[0]}")
            lrs.append(scheduler.get_last_lr()[0])

    plt.figure()
    plt.plot(lrs)
    plt.savefig("./test.png", bbox_inches="tight")
