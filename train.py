import torch.optim as optim
import torch.utils.data
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import json
import numpy as np
from utils.sam_utils import get_sam_args
from models.model_single import ModelEmb
from dataset.glas import get_glas_dataset
from dataset.MoNuBrain import get_monu_dataset
from dataset.polyp import get_polyp_dataset, get_tests_polyp_dataset
from dataset.npydataset import get_npy_dataset
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F
import matplotlib.pyplot as plt


def norm_batch(x):
    bs = x.shape[0]
    Isize = x.shape[-1]
    min_value = x.view(bs, -1).min(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    max_value = x.view(bs, -1).max(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    x = (x - min_value) / (max_value - min_value + 1e-6)
    return x


def Dice_loss(y_true, y_pred, smooth=1):
    alpha = 0.5
    beta = 0.5
    tp = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    fn = torch.sum(y_true * (1 - y_pred), dim=(1, 2, 3))
    fp = torch.sum((1 - y_true) * y_pred, dim=(1, 2, 3))
    tversky_class = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1 - torch.mean(tversky_class)


def get_dice_ji(predict, target):
    predict = predict + 1
    target = target + 1
    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))
    ji = float(np.nan_to_num(tp / (tp + fp + fn)))
    dice = float(np.nan_to_num(2 * tp / (2 * tp + fp + fn)))
    return dice, ji


def open_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    a = os.listdir(path)
    os.mkdir(path + '/gpu' + str(len(a)))
    return str(len(a))


def gen_step(optimizer, gts, masks, criterion, accumulation_steps, step):
    size = masks.shape[2:]
    gts_sized = F.interpolate(gts.unsqueeze(dim=1), size, mode='nearest')
    loss = criterion(masks, gts_sized) + Dice_loss(masks, gts_sized)
    loss.backward()
    if (step + 1) % accumulation_steps == 0:  # Wait for several backward steps
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()


def get_input_dict(imgs, original_sz, img_sz, gts=None):
    batched_input = []
    for i, img in enumerate(imgs):
        input_size = tuple([int(x) for x in img_sz[i].squeeze().tolist()])
        original_size = tuple([int(x) for x in original_sz[i].squeeze().tolist()])
        if gts is not None:
            gt = gts[i]
        else:
            gt = None
        singel_input = {
            'image': img,
            'original_size': original_size,
            'image_size': input_size,
            'point_coords': None,
            'point_labels': None,
            'gt_mask': gt
        }
        batched_input.append(singel_input)
    return batched_input


def postprocess_masks(masks_dict):
    masks = torch.zeros((len(masks_dict), *masks_dict[0]['low_res_logits'].squeeze().shape)).unsqueeze(dim=1).cuda()
    ious = torch.zeros(len(masks_dict)).cuda()
    for i in range(len(masks_dict)):
        cur_mask = masks_dict[i]['low_res_logits'].squeeze()
        cur_mask = (cur_mask - cur_mask.min()) / (cur_mask.max() - cur_mask.min())
        masks[i, 0] = cur_mask.squeeze()
        ious[i] = masks_dict[i]['iou_predictions'].squeeze()
    return masks, ious


def train_single_epoch(ds, model, sam, optimizer, transform, epoch, debug, output_path=None):
    loss_list = []
    pbar = tqdm(ds)
    criterion = nn.BCELoss()
    Idim = int(args['Idim'])
    optimizer.zero_grad()
    for ix, (imgs, gts, original_sz, img_sz) in enumerate(pbar):
        orig_imgs = imgs.to(sam.device)
        gts = gts.to(sam.device)

        if debug:
            # create batch image with GT mask for epoch visualization
            os.makedirs(output_path, exist_ok=True)
            fig, axes = plt.subplots((len(orig_imgs) + 1) // 2, 2, figsize=(10, 5*len(orig_imgs)))
            
            for i, img in enumerate(orig_imgs):
                ax = axes[i // 2, i % 2]
                img = img.squeeze().permute(1, 2, 0).cpu().numpy()
                img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize image values
                gt = gts[i].squeeze().cpu().numpy()
                # draw the segmentation mask in light green
                img[gt > 0] = img[gt > 0] * 0.5 + 0.5 * np.array([0, 1, 0])
                # draw white bounding box around the GT mask
                y_indices, x_indices = np.where(gt > 0)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor='white', facecolor='none')
                ax.add_patch(rect)
                ax.imshow(img)
            # Remove empty subplots
            for i in range(len(orig_imgs), len(axes.flatten())):
                fig.delaxes(axes.flatten()[i])

            img_output_path = os.path.join(output_path, 'batch' + str(ix) + '.png')
            plt.savefig(img_output_path)
            plt.close(fig)

        orig_imgs_small = F.interpolate(orig_imgs, (Idim, Idim), mode='bilinear', align_corners=True)
        dense_embeddings = model(orig_imgs_small)
        batched_input = get_input_dict(orig_imgs, original_sz, img_sz)
        masks = norm_batch(sam_call(batched_input, sam, dense_embeddings))
        loss = gen_step(optimizer, gts, masks, criterion, accumulation_steps=4, step=ix)
        loss_list.append(loss)
        pbar.set_description(
            '(train | {}) epoch {epoch} ::'
            ' loss {loss:.4f}'.format(
                'Medical',
                epoch=epoch,
                loss=np.mean(loss_list)
            ))
    return np.mean(loss_list)


def inference_ds(ds, model, sam, transform, epoch, args, inference_w_gt_as_mask=False, 
                 debug=False, output_dir_path=None):
    pbar = tqdm(ds)
    model.eval()
    iou_list = []
    dice_list = []
    Idim = int(args['Idim'])
    for i, (imgs, gts, original_sz, img_sz) in enumerate(pbar):

        orig_imgs = imgs.to(sam.device)
        gts = gts.to(sam.device)
        orig_imgs_small = F.interpolate(orig_imgs, (Idim, Idim), mode='bilinear', align_corners=True)
        dense_embeddings = model(orig_imgs_small)
        batched_input = get_input_dict(orig_imgs, original_sz, img_sz, gts)
        masks = norm_batch(sam_call(batched_input, sam, dense_embeddings, 
                                    take_gt_as_mask=inference_w_gt_as_mask))
        input_size = tuple([int(x) for x in img_sz[0].squeeze().tolist()])
        original_size = tuple([int(x) for x in original_sz[0].squeeze().tolist()])
        masks = sam.postprocess_masks(masks, input_size=input_size, original_size=original_size)
        gts = sam.postprocess_masks(gts.unsqueeze(dim=0), input_size=input_size, original_size=original_size)
        masks = F.interpolate(masks, (Idim, Idim), mode='bilinear', align_corners=True)
        gts = F.interpolate(gts, (Idim, Idim), mode='nearest')
        masks[masks > 0.5] = 1
        masks[masks <= 0.5] = 0

        if debug and output_dir_path is not None:
            os.makedirs(output_dir_path, exist_ok=True)
            img = orig_imgs_small[0].squeeze().permute(1, 2, 0).cpu().numpy()
            img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize image values
            # create a single image with GT mask for epoch visualization
            fig, ax = plt.subplots(figsize=(10, 10))
            gt = gts[0].squeeze().cpu().numpy()
            # draw the segmentation mask in light green
            img[gt > 0] = img[gt > 0] * 0.5 + 0.5 * np.array([0, 1, 0])
            # draw white bounding box around the GT mask
            y_indices, x_indices = np.where(gt > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor='white', facecolor='none')
            ax.add_patch(rect)
            # draw the segmentation mask in red
            mask = masks[0].squeeze().detach().cpu().numpy()
            img[mask > 0] = img[mask > 0] * 0.5 + 0.5 * np.array([1, 0, 0])
            ax.imshow(img)
            img_output_path = os.path.join(output_dir_path, 'inference_img_' + str(i) + '.png')
            plt.savefig(img_output_path)
            plt.close(fig)

        dice, ji = get_dice_ji(masks.squeeze().detach().cpu().numpy(),
                               gts.squeeze().detach().cpu().numpy())
        iou_list.append(ji)
        dice_list.append(dice)
        pbar.set_description(
            '(Inference | {task}) Epoch {epoch} :: Dice {dice:.4f} :: IoU {iou:.4f}'.format(
                task=args['task'],
                epoch=epoch,
                dice=np.mean(dice_list),
                iou=np.mean(iou_list)))
    return np.mean(iou_list), np.mean(dice_list)


def sam_call(batched_input, sam, dense_embeddings_from_model, take_gt_as_mask=False):
    with torch.no_grad():
        print ("sam_call")
        input_images = torch.stack([sam.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = sam.image_encoder(input_images)
        if take_gt_as_mask:
            input_gts = torch.stack([x["gt_mask"] for x in batched_input], dim=0)
            bboxes = []
            for i in range(input_gts.shape[0]):
                y_indices, x_indices = torch.where(input_gts[i] > 0)
                x_min, x_max = torch.min(x_indices), torch.max(x_indices)
                y_min, y_max = torch.min(y_indices), torch.max(y_indices)
                H, W = input_gts.shape[-2:]
                x_min = torch.clamp(x_min, 0, W)
                x_max = torch.clamp(x_max, 0, W)
                y_min = torch.clamp(y_min, 0, H)
                y_max = torch.clamp(y_max, 0, H)
                bboxes.append([x_min, y_min, x_max, y_max])
            bboxes = torch.tensor(bboxes, device=input_gts.device)
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(points=None, boxes=bboxes, masks=None)
        else:
            sparse_embeddings_none, dense_embeddings_none = sam.prompt_encoder(points=None, boxes=None, masks=None)
            sparse_embeddings = sparse_embeddings_none
            dense_embeddings = dense_embeddings_from_model

    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    return low_res_masks


def main(args=None, sam_args=None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = ModelEmb(args=args).to(device)
    sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
    sam.to(device=device)
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    optimizer = optim.Adam(model.parameters(),
                           lr=float(args['learning_rate']),
                           weight_decay=float(args['WD']))
    if args['task'] == 'monu':
        trainset, testset = get_monu_dataset(args, sam_trans=transform)
    elif args['task'] == 'glas':
        trainset, testset = get_glas_dataset(args, sam_trans=transform)
    elif args['task'] == 'polyp':
        print ("getting polyp dataset")
        trainset, testset = get_polyp_dataset(args, sam_trans=transform)
        print ("got polyp dataset")
    elif args['task'] == 'CT':
        train_data_root = args['train_data_root']   
        test_data_root = args['evaluation_data_root']
        trainset, testset = get_npy_dataset(train_data_root, test_data_root)
    elif args['task'] == 'acdc':
        train_data_root = args['train_data_root']   
        test_data_root = None
        trainset, testset = get_npy_dataset(train_data_root, test_data_root)

    ds = torch.utils.data.DataLoader(trainset, batch_size=int(args['Batch_size']), shuffle=True,
                                     num_workers=int(args['nW']), drop_last=True)
    print ("ds loaded, loading val ds")
    ds_val = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                         num_workers=int(args['nW_eval']), drop_last=False)
    print ("all ds loaded")
    best = 0
    path_best = os.path.join(args['run_output_path'], 'best.csv')
    f_best = open(path_best, 'w')
    print ("start training")
    for epoch in range(int(args['epoches'])):
        single_epoch_output_path = os.path.join(args['run_output_path'], 'epoch' + str(epoch))
        train_single_epoch(ds, model.train(), sam.eval(), optimizer, transform, epoch, 
                           args['debug'], output_path=single_epoch_output_path)
        if ds_val.dataset is not None:
            with torch.no_grad():
                IoU_val, dice_val = inference_ds(ds_val, model.eval(), sam, transform, epoch, args)
                if IoU_val > best:
                    torch.save(model, args['path_best'])
                    best = IoU_val
                    print('best results: ' + str(best))
                    f_best.write(str(epoch) + ',' + str(best) + '\n')
                    f_best.flush()
        else:
            torch.save(model, args['path'])
            # we save the epoch number so we can know where we stopped 
            # eventhough its not the best model
            f_best.write(str(epoch) + ',' + str(0) + '\n')
            f_best.flush()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-lr', '--learning_rate', default=0.0003, help='learning_rate', required=False)
    parser.add_argument('-bs', '--Batch_size', default=3, help='batch_size', required=False)
    parser.add_argument('-epoches', '--epoches', default=5000, help='number of epoches', required=False)
    parser.add_argument('-nW', '--nW', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-nW_eval', '--nW_eval', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-WD', '--WD', default=1e-4, help='evaluation iteration', required=False)
    parser.add_argument('-task', '--task', default='glas', help='evaluation iteration', required=False)
    parser.add_argument('-depth_wise', '--depth_wise', default=False, help='image size', required=False)
    parser.add_argument('-order', '--order', default=85, help='image size', required=False)
    parser.add_argument('-Idim', '--Idim', default=512, help='image size', required=False)
    parser.add_argument('-rotate', '--rotate', default=22, help='image size', required=False)
    parser.add_argument('-scale1', '--scale1', default=0.75, help='image size', required=False)
    parser.add_argument('-scale2', '--scale2', default=1.25, help='image size', required=False)
    parser.add_argument('--debug', action='store_true', help='debug mode', required=False)
    parser.add_argument('--run_name', help='run name, its mendaotry as we need easy way to tell between runs',
                         required=True)
    parser.add_argument('--sam_model_type', type=str,
                    default='vit_b', help='SAM model type', required=False)
    parser.add_argument('--sam_checkpoint_dir_path', type=str, default='cp',
                        help='SAM checkpoint directory path', required=False)
    parser.add_argument('--train_data_root', type=str,
                        help='train data root', required=False, default=None)
    parser.add_argument('--evaluation_data_root', type=str,
                        help='Evaluation data root', required=False, default=None)
    
    args = vars(parser.parse_args())
    run_output_path = os.path.join('results', args['run_name'])
    args['run_output_path'] = run_output_path
    os.makedirs(run_output_path, exist_ok=True)
    # folder = open_folder('results')  # stop with the orig GPU counting. maybe later use.
    # args['folder'] = folder
    args['path'] = os.path.join(run_output_path,
                                'net_last.pth')
    args['path_best'] = os.path.join(run_output_path,
                                     'net_best.pth')
    args['vis_folder'] = os.path.join(run_output_path, 'vis')
    os.makedirs(args['vis_folder'], exist_ok=True)

    sam_args = get_sam_args(sam_model_type=args["sam_model_type"],
                            sam_checkpoint_dir_path=args["sam_checkpoint_dir_path"])

    # save to json file all the info of the current run 
    run_info = {"run_name": args['run_name']}
    run_info['args'] = args
    run_info["sam_args"] = sam_args
    with open(os.path.join(run_output_path, 'run_info.json'), 'w') as f:
        json.dump(run_info, f)
    main(args=args, sam_args=sam_args)

