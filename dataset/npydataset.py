import os
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset

class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = os.path.join(data_root, "gts")
        self.img_path = os.path.join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(os.path.join(self.gt_path, "**/*.npy"), recursive=True)
        )
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
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(
            gt == random.choice(label_ids.tolist())
        )  # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
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
    
def get_npy_dataset(data_root=None, test_data_root=None):
    if data_root:
        ds_train = NpyDataset(data_root)
    else:
        ds_train = None
    if test_data_root:
        ds_test = NpyDataset(test_data_root)
    else:
        ds_test = None
    return ds_train, ds_test

# def copy_only_images_with_gt(data_root):
#     gt_path = os.path.join(data_root, "gts")
#     img_path = os.path.join(data_root, "imgs")
#     gt_path_files = sorted(
#         glob.glob(os.path.join(gt_path, "**/*.npy"), recursive=True)
#     )
#     gt_path_files = [
#         file
#         for file in gt_path_files
#         if os.path.isfile(os.path.join(img_path, os.path.basename(file)))
#     ]
#     print(f"number of images: {len(gt_path_files)}")
    
#     output_dir_path = os.path.join(data_root, 'imgs_with_gt')
#     os.makedirs(output_dir_path, exist_ok=True)
#     for file in tqdm(gt_path_files, desc="Copying images with ground truth"):
#         img_name = os.path.basename(file)
#         orig_img_path = os.path.join(img_path, img_name)
#         output_img_path = os.path.join(output_dir_path, img_name)
#         # copy the image to the new folder
#         os.system(f"cp {orig_img_path} {output_img_path}")

# if __name__ == '__main__':
#     from tqdm import tqdm
#     import argparse

#     parser = argparse.ArgumentParser(description='Description of your program')
#     parser.add_argument('--npy_path', default='data/npy', required=False)
#     args = vars(parser.parse_args())

#     copy_only_images_with_gt(args['npy_path'])