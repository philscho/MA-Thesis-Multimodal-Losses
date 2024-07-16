import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def get_custom_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    initial_lr: float
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return current_step / num_warmup_steps * (1 - initial_lr) + initial_lr
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps))))
    
    return LambdaLR(optimizer, lr_lambda)


# ### From transformers.optimization
# def _get_cosine_schedule_with_warmup_lr_lambda(
#     current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
# ):
#     if current_step < num_warmup_steps:
#         return float(current_step) / float(max(1, num_warmup_steps))
#     progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
#     return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
