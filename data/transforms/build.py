import torchvision.transforms as T

from .transforms import RandomErasing

def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalization(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = T.compose([
            T.RandomResizedCrop(size=cfg.INPUT.SIZE_TRAIN, 
                scale=(cfg.INPUT.MIN_SCALE_TRAIN, cfg.INPUT.MAX_SCALE_TRAIN)
            ), 
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.ToTensor(), 
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])
    return transform