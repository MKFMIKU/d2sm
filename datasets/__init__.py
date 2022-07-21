import importlib
from datasets.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    dataset_filename = "datasets." + dataset_name.lower() + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace("_", "") + "dataset"
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase."
            % (dataset_filename, target_dataset_name)
        )

    return dataset


def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(args, training=True):
    dataset = find_dataset_using_name(args.dataset)
    instance = dataset(args, training)
    print(
        "Training" if training else "Testing",
        "dataset [%s] was created with number [%d]" % (type(instance).__name__, len(instance)),
    )
    return instance
