"""Preprocessing utilities for MuseTalk face detection and landmark extraction."""

import sys
from src.musetalk.utils.face_detection import FaceAlignment, LandmarksType
from os import listdir, path
import subprocess
import numpy as np
import cv2
import pickle
import os
import json
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
import torch
from tqdm import tqdm

# Lazy initialization of models
_model = None
_fa = None

def _get_model():
    """Lazy initialization of mmpose model."""
    global _model
    if _model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_file = './models/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
        checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'
        if os.path.exists(config_file) and os.path.exists(checkpoint_file):
            _model = init_model(config_file, checkpoint_file, device=device)
        else:
            # Fallback: model files not found, will use face detection only
            _model = None
    return _model

def _get_face_alignment():
    """Lazy initialization of face alignment model."""
    global _fa
    if _fa is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)
    return _fa

# Marker if the bbox is not sufficient
coord_placeholder = (0.0, 0.0, 0.0, 0.0)

def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def get_bbox_range(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:', upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    
    model = _get_model()
    fa = _get_face_alignment()
    
    for fb in tqdm(batches):
        if model is not None:
            results = inference_topdown(model, np.asarray(fb)[0])
            results = merge_data_samples(results)
            keypoints = results.pred_instances.keypoints
            face_land_mark = keypoints[0][23:91]
            face_land_mark = face_land_mark.astype(np.int32)
        else:
            # Fallback: use face detection only
            face_land_mark = None
        
        # get bounding boxes by face detection
        bbox = fa.get_detections_for_batch(np.asarray(fb))
        
        # adjust the bounding box refer to landmark
        for j, f in enumerate(bbox):
            if f is None:  # no face in the image
                coords_list += [coord_placeholder]
                continue
            
            if face_land_mark is not None:
                half_face_coord = face_land_mark[29]
                range_minus = (face_land_mark[30] - face_land_mark[29])[1]
                range_plus = (face_land_mark[29] - face_land_mark[28])[1]
                average_range_minus.append(range_minus)
                average_range_plus.append(range_plus)
                if upperbondrange != 0:
                    half_face_coord[1] = upperbondrange + half_face_coord[1]
            else:
                # Use face detection bbox directly
                coords_list += [f]
                continue

    text_range = f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus)) if average_range_minus else 0}~{int(sum(average_range_plus) / len(average_range_plus)) if average_range_plus else 0} ] , the current value: {upperbondrange}"
    return text_range

def get_landmark_and_bbox(img_list, upperbondrange=0):
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:', upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    
    model = _get_model()
    fa = _get_face_alignment()
    
    for fb in tqdm(batches):
        if model is not None:
            results = inference_topdown(model, np.asarray(fb)[0])
            results = merge_data_samples(results)
            keypoints = results.pred_instances.keypoints
            face_land_mark = keypoints[0][23:91]
            face_land_mark = face_land_mark.astype(np.int32)
        else:
            # Fallback: use face detection only
            face_land_mark = None
        
        # get bounding boxes by face detection
        bbox = fa.get_detections_for_batch(np.asarray(fb))
        
        # adjust the bounding box refer to landmark
        for j, f in enumerate(bbox):
            if f is None:  # no face in the image
                coords_list += [coord_placeholder]
                continue
            
            if face_land_mark is not None:
                half_face_coord = face_land_mark[29]
                range_minus = (face_land_mark[30] - face_land_mark[29])[1]
                range_plus = (face_land_mark[29] - face_land_mark[28])[1]
                average_range_minus.append(range_minus)
                average_range_plus.append(range_plus)
                if upperbondrange != 0:
                    half_face_coord[1] = upperbondrange + half_face_coord[1]
                half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
                min_upper_bond = 0
                upper_bond = max(min_upper_bond, half_face_coord[1] - half_face_dist)
                
                f_landmark = (np.min(face_land_mark[:, 0]), int(upper_bond), np.max(face_land_mark[:, 0]), np.max(face_land_mark[:, 1]))
                x1, y1, x2, y2 = f_landmark
                
                if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:  # if the landmark bbox is not suitable, reuse the bbox
                    coords_list += [f]
                    w, h = f[2] - f[0], f[3] - f[1]
                    print("error bbox:", f)
                else:
                    coords_list += [f_landmark]
            else:
                # Use face detection bbox directly
                coords_list += [f]
    
    print("********************************************bbox_shift parameter adjustment**********************************************************")
    print(f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus)) if average_range_minus else 0}~{int(sum(average_range_plus) / len(average_range_plus)) if average_range_plus else 0} ] , the current value: {upperbondrange}")
    print("*************************************************************************************************************************************")
    return coords_list, frames
