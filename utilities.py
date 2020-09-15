import sys
import time
import math
import utils
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
style.use('fivethirtyeight')
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset


def get_object_detection_model(num_classes, trainable_backbone_layers=3):
    """
    Prepares pretrained Faster R-CNN model with a final layer for retraining.

    """
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                 trainable_backbone_layers=trainable_backbone_layers)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the final layer with a new one that will be unfrozen
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ADAPTED FROM https://github.com/pytorch/vision/blob/master/references/detection/engine.py #####
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    loss_tracker = []  # added by Stephen Kaplan
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        loss_tracker.append(loss_value)  # added by Stephen Kaplan

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger, loss_tracker


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def get_validation_loss(model, data_loader, device):
    """Added by Stephen Kaplan"""
    loss_tracker = []
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        loss_tracker.append(loss_value)  # added by Stephen Kaplan

    return loss_tracker


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


def train_model(obj_class_labels, trainable_layers, device, learning_rate, momentum, weight_decay, step_size, gamma,
                num_epochs, data_loader_train, data_loader_val, score_val=False):
    # Load Faster R-CNN Model pretrained on COCO and replace its classifier with a new one that has `num_classes`.
    # add extra class for background
    model = get_object_detection_model(len(obj_class_labels) + 1, trainable_backbone_layers=trainable_layers)

    # move model to the right device
    model.to(device)

    # create optimizer that will only train final layers
    optimizer = torch.optim.SGD(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    training_losses = []
    validation_losses = []
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        metric_log, loss_tracker = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=100)
        training_losses.append(np.mean(loss_tracker))

        # save losses for plotting later
        if score_val:
            validation_loss = get_validation_loss(model, data_loader_val, device)
            validation_losses.append(np.mean(validation_loss))

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        evaluate(model, data_loader_val, device=device)

        # early stopping condition
        if score_val and epoch > 0:
            if validation_losses[-1] > (validation_losses[-2] + 0.01):
                print('Training Stopped. Early stopping criteria met.')
                break

    plt.plot(training_losses)
    if score_val:
        # plot loss curve
        plt.plot(validation_losses)
        plt.legend(['Training Loss', 'Validation Loss'])
    else:
        plt.legend(['Training Loss'])

    plt.title(f'Loss Curve (Learning Rate = {learning_rate})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    return model
