import os
import sys
import glob
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.tfs import get_ct_transform, get_mri_transform

LABELS_METADATA_JSON_FILE_NAME = "labels_metadata.json"



class NpyDataset(Dataset):
    def __init__(self, data_root, augmentations=None, desired_label: int=1, bbox_shift=20):
        self.data_root = data_root
        self.augmentations = augmentations
        self.derired_label = desired_label
        self.gt_path = os.path.join(data_root, "gts")
        self.img_path = os.path.join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(os.path.join(self.gt_path, "**/*.npy"), recursive=True)
        )
        ## we want filter GTs with specific label
        label_json_file_path = os.path.join(data_root, LABELS_METADATA_JSON_FILE_NAME)
        if os.path.isfile(label_json_file_path):
            with open(label_json_file_path, "r") as f:
                labels_metadata = json.load(f)
            self.gt_path_files = [
                file
                for file in self.gt_path_files
                if os.path.basename(file) in labels_metadata[str(desired_label)]
            ]
        else:
            # if the metadata file does not exist, we will filter the files by the desired label on the fly
            self.gt_path_files = [
                file
                for file in self.gt_path_files
                if np.unique(np.load(file)).tolist()[1] in [desired_label]
            ]
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(os.path.join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift


        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            os.path.join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        # now have the image bertween [0, 255] as our model expects
        # this is an addition to the original code
        img_1024 = np.uint8(img_1024 * 255.0)
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        label_ids = np.unique(gt)[1:]
        # random_label = random.choice(label_ids.tolist())
        # print ("random_label: ", random_label)
        gt2D = np.uint8(
            gt == self.derired_label
        )  # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        # apply augmentations
        if self.augmentations:
            img_1024, gt2D = self.augmentations(img_1024, gt2D)

        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        # return (
        #     torch.tensor(img_1024).float(),
        #     torch.tensor(gt2D[None, :, :]).long(),
        #     torch.tensor(bboxes).float(),
        #     img_name,
        # )
        preprocessed_img = torch.tensor(img_1024).float()
        preprocessed_mask = torch.tensor(gt2D).float()
        tensor_original_size = torch.tensor([H, W])
        tensor_image_size = tensor_original_size
        return preprocessed_img, preprocessed_mask, tensor_original_size, tensor_image_size
    
def get_npy_dataset(data_root=None, test_data_root=None, transfrom='ct'):
    if transfrom == 'ct':
        transform_train, transform_test = get_ct_transform()
    elif transfrom == 'mri':
        transform_train, transform_test = get_mri_transform()
    else:
        raise ValueError("Invalid transform type")
    if data_root:
        ds_train = NpyDataset(data_root, augmentations=transform_train)
    else:
        ds_train = None
    if test_data_root:
        ds_test = NpyDataset(test_data_root, transform_test)
    else:
        ds_test = None
    return ds_train, ds_test

def draw_data(data_root):
    import matplotlib.pyplot as plt

    ds_to_draw, ds_none = get_npy_dataset(data_root=data_root)
    ds = torch.utils.data.DataLoader(ds_to_draw, batch_size=1, shuffle=True,
                                     num_workers=1, drop_last=True)
    pbar = tqdm(ds)
    for i, (imgs, gts, original_sz, img_sz) in enumerate(pbar):
        print ("imgs shape: ", imgs.shape, "gts shape: ", gts.shape, "original_sz: ", original_sz, "img_sz: ", img_sz)
        img = imgs.squeeze().permute(1, 2, 0).cpu().numpy()
        img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize image values
        gt = gts[0].squeeze().cpu().numpy()
        plt.imshow(img, cmap='gray')
        plt.imshow(gt, cmap='Reds', alpha=0.5)
        plt.title(f" {i+1} of {len(pbar)}")
        plt.show()

def create_labels_metadata(data_root):
    """
    this function creates a metadata file for the labels in the dataset,
    i.e it outputs a file so that for each label in the dataset, we have filenames in which that label appears
    """
    from collections import defaultdict

    data_root = data_root
    gt_path = os.path.join(data_root, "gts")
    img_path = os.path.join(data_root, "imgs")
    gt_path_files = sorted(
        glob.glob(os.path.join(gt_path, "**/*.npy"), recursive=True)
    )
    gt_path_files = [
        file
        for file in gt_path_files
        if os.path.isfile(os.path.join(img_path, os.path.basename(file)))
    ]
    pbar = tqdm(gt_path_files)
    labels_metadata = defaultdict(list)
    for i, gt_file_path in enumerate(pbar):
        gt = np.load(gt_file_path, "r", allow_pickle=True)
        label_ids = np.unique(gt)[1:]
        for label in label_ids:
            labels_metadata[label].append(os.path.basename(gt_file_path))
    # have labels_metadata so keys sorted by integert then convert to string
    sorted_labels_metadata = {str(k): v for k, v in dict(sorted(labels_metadata.items())).items()}
    file_output_path = os.path.join(data_root, LABELS_METADATA_JSON_FILE_NAME)
    with open(file_output_path, "w") as f:
        json.dump(sorted_labels_metadata, f)




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        help='data root', required=False, default='/media/tal/tal_ssd/datasets/npy/CT_Abd')
    parser.add_argument('--application', type=str,
                        help='application: draw_data / create_labels_metadata', required=False, default='draw_data')
    parser.add_argument('--desired_label', type=int,
                        help='desired label', required=False, default=1)
    parser.add_argument('--debug', action='store_true', help='debug mode', required=False)

    return parser.parse_args()

if __name__ == '__main__':
    from tqdm import tqdm
    import argparse

    parser = argparse.ArgumentParser()

    args = parse_args()
    if args.application == 'draw_data':
        draw_data(args.data_root)
    elif args.application == 'create_labels_metadata':
        create_labels_metadata(args.data_root)


