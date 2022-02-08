from .transforms import get_transform
from .cubox import CUBOXdataset
from torch.utils.data import ConcatDataset, DataLoader
import tqdm


def get_cubox_dataset(data_config, data_root, transform, split, combine=True, dataset_name='cubox'):
    datasets = []
    for occl in data_config[split]:
        datasets.append(CUBOXdataset(data_root, split=split, occlusion=occl, transform=transform))
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)
    else:
        dataset = datasets[0]
    print(f"Total {len(dataset)} number of samples loaded")
    classes = datasets[0].classes
    return dataset, classes

def get_test_dataset(data_config, data_root, transform_type, dataset_name='cubox'):
    _, val_transform = get_transform(transform_type)
    return get_cubox_dataset(data_config, data_root, val_transform, 'test', combine=True, dataset_name=dataset_name)


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


def get_test_loader(data_config, data_root, transform_type, args):
    print("Preparing test datasets...")
    test_dataset, classes = get_test_dataset(data_config, data_root, transform_type)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True)
    print('Test loaders prepared!')
    return test_loader, classes