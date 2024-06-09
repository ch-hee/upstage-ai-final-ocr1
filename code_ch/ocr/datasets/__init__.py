from .base import OCRDataset
from .db_collate_fn import DBCollateFN
from .transforms import DBTransforms
from hydra.utils import instantiate


def get_datasets_by_cfg(config):
    train_dataset1 = instantiate(config.train_dataset1)
    # train_dataset2 = instantiate(config.train_dataset2)
    # train_dataset3 = instantiate(config.train_dataset3)
    # train_dataset = train_dataset1 + train_dataset2 + train_dataset3

    val_dataset = instantiate(config.val_dataset)
    test_dataset = instantiate(config.test_dataset)
    predict_dataset = instantiate(config.predict_dataset)
    return {
        'train': train_dataset1,
        'val': val_dataset,
        'test': test_dataset,
        'predict': predict_dataset
    }
