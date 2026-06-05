import torch.optim as optim
from transformers import get_linear_schedule_with_warmup

def get_optimizer_and_scheduler(model, lr: float = 1e-4, weight_decay: float = 0.01, num_warmup_steps: int = 100, num_training_steps: int = 1000):
    """
    Configure AdamW optimizer and learning rate scheduler.
    """
    # Exclude bias and LayerNorm parameters from weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return optimizer, scheduler
