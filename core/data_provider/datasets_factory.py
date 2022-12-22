from torchvision import transforms
from torch.utils.data import DataLoader


def data_provider(dataset, configs, data_train_path, data_test_path, batch_size,
                  is_training=True,
                  is_shuffle=True):
    if dataset == 'mnist':
        from core.data_provider.mm import MovingMNIST

    if is_training:
        mode = 'train'
        num_workers = configs.num_workers
        root = 'datasets/mnist/'
    else:
        mode = 'test'
        num_workers = 0
        root = 'datasets/mnist/'
    # else:
    dataset = MovingMNIST(is_train=is_training,
                            root=root,
                            n_frames=20,
                            num_objects=[2])
    return DataLoader(dataset,
                      pin_memory=True,
                      batch_size=batch_size,
                      shuffle=is_shuffle,
                      num_workers=num_workers)
