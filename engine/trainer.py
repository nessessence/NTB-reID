import logging 

import torch
from torch.nn import nn

from utils.reid_metric import R1_mAP

def train_step(inputs, labels, model, optimizer, loss_fn):
    preds = model(inputs)
    loss = loss_fn(preds, labels)
    return loss, preds 
    
def test_step(inputs, labels, model, optimizer, loss_fn): 
    with torch.no_grad():
        preds = model(inputs)
        loss = loss_fn(preds, labels)
    return loss, preds 

def do_train(
    cfg, 
    model, 
    train_loader,
    val_loader,
    optimizer,
    scheduler, 
    loss_fn
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("template_model.train")
    logger.info("Start training")
    # trainer = 
    # evaluator = 
    # checkpointer = 
    
    