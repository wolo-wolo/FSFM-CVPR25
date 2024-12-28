# -*- coding: utf-8 -*-
# Author: Gaojian Wang@ZJUICSR
# --------------------------------------------------------
# This source code is licensed under the Attribution-NonCommercial 4.0 International License.
# You can find the license in the LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import cv2
import os
import sys
import numpy as np
from config import cfg
import dlib
from skimage import transform as trans
import multiprocessing


# Detect face with dlib
face_detector = dlib.get_frontal_face_detector()


def get_boundingbox(face, width, height, minsize=None):
    """
    From FF++:
    https://github.com/ondyari/FaceForensics/blob/master/classification/detect_from_video.py
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param cfg.face_scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * cfg.face_scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def get_keypts(image, face, predictor):
    # detect the facial landmarks for the selected face
    shape = predictor(image, face)

    # select the key points for the eyes, nose, and mouth
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)

    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

    return pts


def img_align_crop(img, landmark=None, outsize=None, scale=1.0):
    """
    align and crop the face according to the given bbox and landmarks
    landmark: 5 key points
    """
    target_size = [112, 112]
    dst = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)

    if target_size[1] == 112:
        dst[:, 0] += 8.0

    dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
    dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

    target_size = outsize

    margin_rate = scale - 1
    x_margin = target_size[0] * margin_rate / 2.
    y_margin = target_size[1] * margin_rate / 2.
    # move
    dst[:, 0] += x_margin
    dst[:, 1] += y_margin
    # resize
    dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
    dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

    src = landmark.astype(np.float32)

    # use skimage tranformation
    tform = trans.SimilarityTransform()
    tform.estimate(src, dst)
    M = tform.params[0:2, :]

    img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

    if outsize is not None:
        img = cv2.resize(img, (outsize[1], outsize[0]), interpolation=cv2.INTER_CUBIC)

    return img


def extract_align_and_save_face(frame, dst_path, save_img_name):
    # Convert to rgb for dlib
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb, 1)
    if len(faces) > 0:
        # For now only take the biggest face
        face = faces[0]
        # Face crop and rescale(follow FF++)
        x, y, size = get_boundingbox(face, frame.shape[1], frame.shape[0])
        rescaled_face = dlib.rectangle(x, y, x + size, y + size)

        predictor_path = 'tools/shape_predictor_81_face_landmarks.dat'
        # Check if predictor path exists
        if not os.path.exists(predictor_path):
            print(f"Predictor path does not exist: {predictor_path}")
            sys.exit()
        face_predictor = dlib.shape_predictor(predictor_path)

        # Get the landmarks/parts for the face in box d only with the five key points
        landmarks = get_keypts(rgb, rescaled_face, face_predictor)
        cropped_face = img_align_crop(rgb, landmarks)
        if len(face_detector(cropped_face, 1)) > 0:
            cv2.imwrite(os.path.join(dst_path, save_img_name), cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))


def extract_and_save_face(frame, dst_path, save_img_name):
    # Convert to rgb for dlib
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb, 1)
    if len(faces) > 0:
        # For now only take the biggest face
        face = faces[0]
        # Face crop and rescale(follow FF++)
        x, y, size = get_boundingbox(face, frame.shape[1], frame.shape[0])
        # Get the landmarks/parts for the face in box d only with the five key points
        cropped_face = rgb[y:y + size, x:x + size]
        # cropped_face = cv2.resize(cropped_face, (cfg.face_size, cfg.face_size), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(dst_path, save_img_name), cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))


def get_frame_index_uniform_sample(total_frame_num, extract_frame_num):
    interval = np.linspace(0, total_frame_num - 1, num=extract_frame_num, dtype=int)
    return interval.tolist()


def extract_face_from_fixed_num_frames(src_video, dst_path, video_name, num_frames=None, align=True):
    """
    1) extract specific num of frames from videos in [1st(index 0) frame, last frame] with uniform sample interval
    2) extract face from frame with specific enlarge size
    """
    video_capture = cv2.VideoCapture(src_video)
    total_frames = video_capture.get(7)

    # extract from the 1st(index 0) frame
    if num_frames is not None:
        frame_indices = get_frame_index_uniform_sample(total_frames, num_frames)
    else:
        frame_indices = range(int(total_frames))

    extract_func = extract_align_and_save_face if align else extract_and_save_face

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for frame_index in frame_indices:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = video_capture.read()
        if not ret:
            continue
        save_img_name = f"{video_name}_frame_{frame_index}{cfg.img_format}"
        pool.apply_async(extract_func, args=(frame, dst_path, save_img_name))

    pool.close()
    pool.join()
    video_capture.release()
    cv2.destroyAllWindows()
