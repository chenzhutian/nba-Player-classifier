import json
import os
import random
import shutil
from predictor import get_predictor
from yolox.tracking_utils.timer import Timer
import cv2
import numpy as np

def get_gt_by_frame(bbox_file: str):
    gtByFrames = {}
    # convert to List[bboxes, List[int]]
    with open(bbox_file) as f:
        annot = json.load(f)
    labels = annot['labels']
    infos = annot['info']
    frames_map = [int(u.split('_')[-1].split('.')[0]) for u in infos['url']]
    for pIdx, player in enumerate(labels):
        frames = player['data']['frames']
        #group by frame
        for frame in frames:
            frame_idx = frames_map[frame['frame']]
            if frame_idx not in gtByFrames:
                gtByFrames[frame_idx] = [[], []]
            gtByFrames[frame_idx][0].append(frame['points'] + [1])
            gtByFrames[frame_idx][1].append(pIdx)
    for k, v in gtByFrames.items():
        gtByFrames[k] = (np.array(v[0]), np.array(v[1]))
    
    return gtByFrames

def ensure_folder(path: str):
    try:
        os.makedirs(path)
    except: pass

def remove_folder(path:str):
    try:
        shutil.rmtree(path)
    except: pass

if __name__ == '__main__':
    gt_labels = os.listdir('./input/bboxes')
    detector = get_predictor()
    OUTPUT_ROOT = './output'
    remove_folder(OUTPUT_ROOT)
    ensure_folder(OUTPUT_ROOT)
    splits = ['train', 'val', 'test']
    for split in splits:
        ensure_folder(OUTPUT_ROOT + f'/{split}/negative')
        ensure_folder(OUTPUT_ROOT + f'/{split}/positive')

    for gt_label_file in gt_labels:
        gt_bboxes = get_gt_by_frame(f'./input/bboxes/{gt_label_file}')
        video_id = gt_label_file.split('.')[0]
        print(video_id)
        
        if not os.path.exists(f'./input/images/{video_id}'):
            continue
        
        # extract positive patches
        for frame_idx, (player_bboxes, player_ids) in gt_bboxes.items():
            img = cv2.imread(f'./input/images/{video_id}/{frame_idx}.jpg')
            img_masked = img.copy()
            for bbox in player_bboxes:
                x1, y1, x2, y2, _ = bbox.astype(int)
                img_masked[y1:y2, x1:x2] = 0
            cv2.imwrite(OUTPUT_ROOT + '/masked.png', img_masked)
            outputs, img_info = detector.inference(img_masked[:, :, :3], Timer())

            output_results = outputs[0]
            if output_results is None: break
            imgH, imgW = img_info['height'], img_info['width']

            # human_bboxes = []
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            human_bboxes = output_results[:, :4]  # x1y1x2y2
            remain_indx = scores > 0.6
            scores = scores[remain_indx]
            human_bboxes = human_bboxes[remain_indx]

            img_size = (800, 1440)
            scale = min(img_size[0] / float(imgH), img_size[1] / float(imgW))
            human_bboxes /= scale
    
            # negative samples
            negative_samples = [(bIdx, b) for bIdx, b in enumerate(human_bboxes) if b.min() > 0]
            negative_samples = random.sample(negative_samples, min(20, len(negative_samples)))
            random.shuffle(negative_samples)
            train_split_idx = int(len(negative_samples) * 0.7)
            val_split_idx = int(len(negative_samples) * 0.8)
            for idx, (bIdx, bbox) in enumerate(negative_samples):
                split = 'train' if idx < train_split_idx else ('val' if idx < val_split_idx else 'test')
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.imwrite(f'{OUTPUT_ROOT}/{split}/negative/{video_id}_{frame_idx}_{bIdx}.png', img[y1:y2, x1:x2])
            
            positive_samples = list(zip(player_ids, player_bboxes))
            random.shuffle(positive_samples)
            train_split_idx = int(len(positive_samples) * 0.7)
            val_split_idx = int(len(positive_samples) * 0.8)
            for idx, (player_id, player_bboxe) in enumerate(positive_samples):
                split = 'train' if idx < train_split_idx else ('val' if idx < val_split_idx else 'test')
                x1, y1, x2, y2, _ = player_bboxe.astype(int)
                x1 = max(x1, 0)
                try:
                    cv2.imwrite(f'{OUTPUT_ROOT}/{split}/positive/{video_id}_{frame_idx}_{player_id}.png', img[y1:y2, x1:x2])
                except:
                    print(f'{OUTPUT_ROOT}/{split}/positive/{video_id}_{frame_idx}_{player_id}.png', x1, y1, x2, y2)
