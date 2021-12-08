from datasets.cifar_dataset import CIFAR10, CIFAR100
from datasets.data import CIFAR10_loader_half, CIFAR10_loader_mix, CIFAR100_loader_half, CIFAR100_loader_mix


def test_cifar10_dataset():
    """
    test cifar 10 dataset 
    """

    cifar10_train = CIFAR10('train')

    print(cifar10_train.class_to_idx)
    for idx, data in enumerate(cifar10_train):

        print(idx)
        print(data[0].shape)
        print(data[1])

    cifar10_test = CIFAR10('test')

    for idx, data in enumerate(cifar10_test):

        print(idx)
        print(data[0].shape)
        print(data[1])


def test_cifar100_dataset():
    """
    test cifar 100 dataset 
    """

    cifar100_train = CIFAR100('train')
    print(cifar100_train.class_to_idx)

    for idx, data in enumerate(cifar100_train):

        print(idx)
        print(data[0].shape)
        print(data[1])

    cifar100_test = CIFAR100('test')

    for idx, data in enumerate(cifar100_test):

        print(idx)
        print(data[0].shape)
        print(data[1])


def test_cifar10_load_half():
    """
    test cifar 10 dataset loader for half labeled or unlabeled data 
    """

    half_loader = CIFAR10_loader_half(batch_size=1, target_list=range(5,10))

    for img, target in half_loader:

        print(img.shape)
        print(target)

def test_cifar10_load_mix():
    """
    test cifar 10 dataset loader for half labeled and unlabeled data 
    """

    mix_loader = CIFAR10_loader_mix(batch_size=2)

    for img, target in mix_loader:
        print(img.shape)
        print(target)

def test_cifar100_load_half():
    """
    test cifar 100 dataset loader for half labeled or unlabeled data 
    """

    half_loader = CIFAR100_loader_half(batch_size=1, target_list=range(50,100))

    for img, target in half_loader:

        print(img.shape)
        print(target)

def test_cifar100_load_mix():
    """
    test cifar 100 dataset loader for half labeled and unlabeled data 
    """

    mix_loader = CIFAR100_loader_mix(batch_size=2)

    for img, target in mix_loader:
        print(img.shape)
        print(target)

if __name__ == "__main__":

    # test_cifar10_dataset()
    # test_cifar100_dataset()

    # test_cifar10_load_half()
    # test_cifar10_load_mix()

    # test_cifar100_load_half()
    # test_cifar100_load_mix()