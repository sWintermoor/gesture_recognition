from tqdm import tqdm
import pickle
import os
import warnings
import sys

from torch.utils.data import random_split

class DatasetHandler:
    def __init__(self, confg):
        self._confg = confg

    # Public
    def process(self, datasets):
        datasets = self._extract_datasets(datasets)
        for dataset_name, dataset_config in datasets.items():
            dataset = dataset_config["dataset"]
            processor = dataset_config["processor"]
            save_file = dataset_config["save_directory"]

            if not os.path.exists(save_file):
                N_V = self.dataset.size()
                ds = []

                for i in tqdm(range(N_V)):
                    for j in range(sys.maxsize):
                        try:
                            frames, label, ID = dataset.__getitem__(i, j)

                            keypoint_sequence = processor.process_video(frames)

                            ds.append([keypoint_sequence, dataset.label_to_index[label]]) # using ID as label
                        except:
                            break

                #Saving new dataset
                with open(save_file, "wb") as f:
                    pickle.dump(ds, f)
            else:
                warnings.warn(f"File for {dataset_name} already exists.")

            with open(save_file, "rb") as f:
                ds = pickle.load(f)
            self._config["processed_dataset"] = ds
            print(f"Successfull processed {dataset_name} dataset")

    def print_statistics(self, datasets):
        datasets = self._extract_datasets(datasets)
        for dataset_name, dataset_config in datasets.items():
            dataset = dataset_config["dataset"]
            processed_dataset = dataset_config["processed_dataset"]

            print(f"Dataset: {dataset_name}")
            print(f"Amount of labels: {len(dataset.get_label_id_dict())}")
            print(f"Amount of videos: {len(dataset.get_video_info())}")

            if processed_dataset:
                print(f"Handler catched processed dataset.")
            else:
                warnings.warn("Handler didn't catch processed version!")

    def get_split_ds(self, dataset_name, train_size, val_size, test_size, shuffle=True):
        try:
            ds = self._confg[dataset_name]["processed_dataset"]
            if shuffle:
                train_ds, val_ds, test_ds = random_split(ds, [train_size, val_size, test_size])
            else:
                train_ds = ds[:train_size]
                val_ds = ds[train_size : (train_size+val_size)]
                test_ds = ds[(train_size+val_size):]

            return train_ds, val_ds, test_ds
        except:
            raise "Ooops! Something went wrong..." #TODO: Better exception
    
    #Private
    def _extract_datasets(self, datasets):
        active_datasets = {}
        for dataset in datasets:
            for name, config in self._confg.items():
                if dataset == name:
                    active_datasets[name] = config

        return active_datasets



    
