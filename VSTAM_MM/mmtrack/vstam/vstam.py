import torch
from torch.utils.data import DataLoader
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.structures.image_list import ImageList
from detection_models.rcnn_headed_model import *
from detectron2.engine import DefaultTrainer
from .data.dataset import VSTAMDataset

def test(config_file,output_file):
    BATCH_SIZE = 1
    SEQUENCE_LENGTH = 5
    CH_IN = 3

    # Here load test dataset
    imgs_suplementary = torch.ones((BATCH_SIZE, SEQUENCE_LENGTH, CH_IN, *cfg.INPUT.MIN_SIZE_TEST)).to("cuda")
    imgs = torch.ones((BATCH_SIZE, CH_IN, *cfg.INPUT.MIN_SIZE_TEST)).to("cuda")
    imgs = ImageList(imgs, [cfg.INPUT.MIN_SIZE_TEST for i in range(BATCH_SIZE)])

    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    model = build_model(cfg)
    model.eval()
    features = model.backbone(imgs, imgs_suplementary=imgs_suplementary)
    proposals, _ = model.proposal_generator(imgs, features)
    instances, _ = model.roi_heads(imgs, features, proposals)
    print(instances)

    with open(output_file,"wb") as fp:
        fp.write(instances)

def train(config_file):

    cfg = get_cfg()
    cfg.merge_from_file(config_file)

    train_videos = cfg.DATASET.TRAIN_VIDEOS
    train_masks = cfg.DATASET.TRAIN_MASKS
    exponential_frames_count = cfg.DATASET.EXPONENTIAL_FRAMES_COUNT
    additional_by_score_count = cfg.DATASET.ADDITIONAL_BY_SCORE_COUNT
    previous_shot_count = cfg.DATASET.PREVIOUS_SHOT_COUNT

    dataset = VSTAMDataset(
        train_videos=train_videos,
        train_masks=train_masks,
        exponential_frames_count=exponential_frames_count,
        additional_by_score_count=additional_by_score_count,
        previous_shot_count=previous_shot_count
    )

    train_dataloader = DataLoader(dataset, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True)

    from detectron2.data import DatasetCatalog, MetadataCatalog
    DatasetCatalog.register("vstam_dataset_train", train_dataloader)
    MetadataCatalog.get("vstam_dataset_train").set(thing_classes=["class1", "class2", ...])

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()