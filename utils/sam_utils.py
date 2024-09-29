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