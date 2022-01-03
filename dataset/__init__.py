from .transforms import DummyTransform, get_transform
from .cubox import CUBOXdataset, occlusion_types
from torch.utils.data import ConcatDataset, DataLoader
from icecream import icecream

def get_dataset(dataset_name, data_config, data_root, transform_type):
    train_transform, val_transform = get_transform(transform_type)
    if dataset_name == "cubox":
        trainsets, valsets, testsets = [], [], []
        for occl in data_config['train']:
            trainsets.append(
                CUBOXdataset(data_root, split='train', occlusion=occl, transform=train_transform))
        trainset = ConcatDataset(trainsets)
        for occl in data_config['val']:
            valsets.append(CUBOXdataset(data_root, split='validation', occlusion=occl, transform=val_transform))
        for occl in data_config['test']:
            testsets.append(CUBOXdataset(data_root, split='test', occlusion=occl, transform=val_transform))
        num_classes = testsets[0].num_classes
    else:
        raise NotImplementedError

    return trainset, valsets, testsets, num_classes


def get_loaders(dataset_name, data_config, data_root, transform_type, args):
    trainset, valsets, testsets, num_classes = get_dataset(dataset_name, data_config, data_root, transform_type)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    print('train loaders prepared')
    val_loaders = []
    for valset in valsets:
        val_loaders.append(DataLoader(valset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False))
    print('Validation loaders prepared')
    test_loaders = []
    for testset in testsets:
        test_loaders.append(DataLoader(testset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False))
    print('Test loaders prepared')

    return train_loader, val_loaders, test_loaders, num_classes
