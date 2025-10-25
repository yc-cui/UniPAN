import torch.nn.functional as F

def l1_loss(pred, gt, split="train"):
    loss = F.l1_loss(pred, gt)
    log_dict = {
        f"{split}/l1_loss": loss
    }
    return loss, log_dict
