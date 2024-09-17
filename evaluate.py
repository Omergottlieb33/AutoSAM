import os
import tqdm
import json
import torch
import argparse
import numpy as np
from typing import Dict
import torch.utils.data
import torch.nn.functional as F

from train import inference_ds
from models.model_single import ModelEmb
from dataset.glas import get_glas_dataset
from dataset.polyp import get_polyp_dataset
from dataset.MoNuBrain import get_monu_dataset
from dataset.npydataset import get_npy_dataset
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


def evaluate(task_to_eval: str, model_path: str, sam_args: Dict,
             evaluation_data_root: str = None,
             num_workers: int = 1, idim: int = 512,
             output_dir: str = 'results/evaluation',
             eval_name: str = 'evaluation',
             debug=False,
             inference_w_gt_as_mask=False):
    
    # prepare output directory and file name
    current_evaluation_output_dir = os.path.join(output_dir, eval_name)
    os.makedirs(current_evaluation_output_dir, exist_ok=True)
    # filename should by eval_name.json, if elready exists, create another file with a number suffix
    json_file_name = f"{eval_name}.json"
    existing_files = [f for f in os.listdir(
        current_evaluation_output_dir) if f.startswith(eval_name) and f.endswith('.json')]
    if len(existing_files) > 0:
        file_num = len(existing_files) + 1
        json_file_name = f"{eval_name}_{file_num}.json"
    json_results_path = os.path.join(current_evaluation_output_dir, json_file_name)
    eval_args = {
        "eval_name": eval_name,
        "task": task_to_eval,
        "evaluation_data_root": evaluation_data_root,
        "model_path": model_path,
        "sam_args": sam_args,
        "num_workers": num_workers,
        "idim": idim,
        "output_dir": output_dir,
        "debug": debug,
        "inference_w_gt_as_mask": inference_w_gt_as_mask
    }

    inference_args = {"Idim": idim,
            "task": task_to_eval,
            "depth_wise": False,  
            "order": 85}
    
    pre_results_json = {
        "eval_args": eval_args,
        "inference_args": inference_args,
        "results": "WIP"
    }
    with open(json_results_path, "w") as f:
        json.dump(pre_results_json, f)


    
    model = ModelEmb(args=inference_args).to(device=device)
    model1 = torch.load(model_path)
    model.load_state_dict(model1.state_dict())
    sam = sam_model_registry[sam_args['model_type']](
        checkpoint=sam_args['sam_checkpoint'])
    sam.to(device=device)
    sam.eval()
    transform = ResizeLongestSide(sam.image_encoder.img_size)

    if task_to_eval == 'monu':
        trainset, testset = get_monu_dataset(args, sam_trans=transform)
        evaluation_data_root = 'MoNuSeg'
    elif task_to_eval == 'glas':
        trainset, testset = get_glas_dataset(sam_trans=transform)
        evaluation_data_root = 'Warwick/'
    elif task_to_eval == 'polyp':
        trainset, testset = get_polyp_dataset(args, sam_trans=transform)
        evaluation_data_root = 'polyp'
    elif task_to_eval == 'FLARE':
        trainset, testset = get_npy_dataset(data_root=None,
                                            test_data_root=evaluation_data_root)
    ds_val = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                         num_workers=num_workers, drop_last=False)

    with torch.no_grad():
        model.eval()
        IoU_val = inference_ds(ds=ds_val, model=model, sam=sam, transform=transform, 
                               epoch=0, args=inference_args, 
                               debug=debug, output_dir_path=current_evaluation_output_dir,
                               inference_w_gt_as_mask=inference_w_gt_as_mask)
        print("evaluated IoU: ", IoU_val)

    # save results
    results_json = {
        "results": {
            "IoU": IoU_val
        },
        "eval_args": eval_args,
        "inference_args": inference_args
    }
    with open(json_results_path, "w") as f:
        json.dump(results_json, f)


def get_sam_args(sam_model_type: str = "vit_b",
                 sam_checkpoint_dir_path: str = "cp"):
    if sam_model_type == "vit_b":
        sam_args = {
            'sam_checkpoint': f"{sam_checkpoint_dir_path}/sam_vit_b.pth",
            'model_type': "vit_b",
            'generator_args': {
                'points_per_side': 8,
                'pred_iou_thresh': 0.95,
                'stability_score_thresh': 0.7,
                'crop_n_layers': 0,
                'crop_n_points_downscale_factor': 2,
                'min_mask_region_area': 0,
                'point_grids': None,
                'box_nms_thresh': 0.7,
            },
            'gpu_id': 0,
        }
    elif sam_model_type == "vit_l":
        sam_args = {
            'sam_checkpoint': f"{sam_checkpoint_dir_path}/sam_vit_l.pth",
            'model_type': "vit_l",
            'generator_args': {
                'points_per_side': 8,
                'pred_iou_thresh': 0.95,
                'stability_score_thresh': 0.7,
                'crop_n_layers': 0,
                'crop_n_points_downscale_factor': 2,
                'min_mask_region_area': 0,
                'point_grids': None,
                'box_nms_thresh': 0.7,
            },
            'gpu_id': 0,
        }
    elif sam_model_type == "vit_h":
        sam_args = {
            'sam_checkpoint': f"{sam_checkpoint_dir_path}/sam_vit_h.pth",
            'model_type': "vit_h",
            'generator_args': {
                'points_per_side': 8,
                'pred_iou_thresh': 0.95,
                'stability_score_thresh': 0.7,
                'crop_n_layers': 0,
                'crop_n_points_downscale_factor': 2,
                'min_mask_region_area': 0,
                'point_grids': None,
                'box_nms_thresh': 0.7,
            },
            'gpu_id': 0,
        }
    else:
        raise ValueError(f"Unknown sam_model_type: {sam_model_type}")
    return sam_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_name', type=str, default='evaluation',
                        help='Evaluation name', required=False)
    parser.add_argument('--task', type=str, default='polyp',
                        help='Task to evaluate', required=True)
    parser.add_argument('--evaluation_data_root', type=str,
                        help='Evaluation data root', required=False)
    parser.add_argument('--model_path', type=str, required=True, help='Path to model')
    parser.add_argument('--sam_model_type', type=str,
                        default='vit_b', help='SAM model type', required=False)
    parser.add_argument('--sam_checkpoint_dir_path', type=str, default='cp',
                        help='SAM checkpoint directory path', required=False)
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers', required=False)
    parser.add_argument('--idim', type=int, default=512,
                        help='Idim', required=False)
    parser.add_argument('--inference_w_gt_as_mask', action='store_true', 
                        help='inference_w_gt_as_mask', required=False)
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                        help='Output directory', required=False)
    parser.add_argument('--debug', action='store_true', help='debug mode', required=False)

    return parser.parse_args()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parse_args()

    eval_name = args.eval_name
    task_to_eval = args.task
    evaluation_data_root = args.evaluation_data_root
    model_path = args.model_path
    sam_model_type = args.sam_model_type
    sam_checkpoint_dir_path = args.sam_checkpoint_dir_path
    num_workers = args.num_workers
    idim = args.idim
    output_dir = args.output_dir
    debug = args.debug
    inference_w_gt_as_mask = args.inference_w_gt_as_mask

    sam_args = get_sam_args(sam_model_type=sam_model_type,
                            sam_checkpoint_dir_path=sam_checkpoint_dir_path)
    
    evaluate(task_to_eval=task_to_eval, model_path=model_path, evaluation_data_root=evaluation_data_root,
             sam_args=sam_args, num_workers=num_workers, idim=idim, output_dir=output_dir, eval_name=eval_name,
             debug=debug, inference_w_gt_as_mask=inference_w_gt_as_mask)
