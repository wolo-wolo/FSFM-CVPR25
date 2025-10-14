import os
import re
import logging
from typing import List
from PIL import Image
import numpy as np
import h5py
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from config import cfg


class RemoveBlackBorders(object):
    def __call__(self, im):
        if type(im) == list:
            return [self.__call__(ims) for ims in im]
        V = np.array(im)
        V = np.mean(V, axis=2)
        X = np.sum(V, axis=0)
        Y = np.sum(V, axis=1)
        y1 = np.nonzero(Y)[0][0]
        y2 = np.nonzero(Y)[0][-1]

        x1 = np.nonzero(X)[0][0]
        x2 = np.nonzero(X)[0][-1]
        return im.crop([x1, y1, x2, y2])

    def __repr__(self):
        return self.__class__.__name__


def ensure_dir_exists(path: str) -> None:
    """Ensure the directory exists."""
    os.makedirs(path, exist_ok=True)


def process_image(img_path: str or Image.Image, output_path: str, remove_black_borders: RemoveBlackBorders) -> None:
    """Process and save an image in the WCS protocol."""
    try:
        # Check if img_path is a file path or an Image object
        if isinstance(img_path, Image.Image):
            img = img_path
        else:
            with Image.open(img_path) as img:
                img = img.copy()  # Ensure the image is not closed after the 'with' block

        processed_img = remove_black_borders(img)
        ensure_dir_exists(os.path.dirname(output_path))
        processed_img.save(output_path)
        # logging.info(f"Processed and saved: {output_path}")
    except Exception as e:
        logging.error(f"Error processing {img_path}: {e}")


def sample_frames(folder_path: str, num_samples: int = 10) -> List[str]:
    """Sample frames for the cefa dataset."""
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])
    if len(files) < num_samples:
        raise ValueError(f"Insufficient number of files in {folder_path} to sample {num_samples} frames.")
    step = len(files) // num_samples
    return [files[i * step] for i in range(num_samples)]


def run_cefa(rootpath: str, output_folder: str, sample_split_file: List[str]) -> None:
    """Process CeFA dataset."""
    remove_black_borders = RemoveBlackBorders()

    for file_path in sample_split_file:
        with open(file_path, "r") as file:
            sample_list = [line.strip() for line in file.readlines()]

        for sample in tqdm(sample_list, desc=f"Processing {file_path}"):
            item = sample.split("/")[-1]
            race = item.split("_")[0]
            subfolder = {"1": "AF", "2": "CA", "3": "EA"}.get(race)

            source_dir = os.path.join(
                rootpath, subfolder, f"{subfolder}-{item.split('_')[1]}", "_".join(item.split("_")[:-1]), "profile"
            )

            if not os.path.exists(source_dir):
                logging.warning(f"Source directory does not exist: {source_dir}")
                continue

            try:
                sampled_frames = sample_frames(source_dir)
                selected_frame = sampled_frames[int(item.split("_")[-1][:2])]
                selected_frame_path = os.path.join(source_dir, selected_frame)

                target_path = os.path.join(output_folder, sample)
                process_image(selected_frame_path, target_path, remove_black_borders)
            except Exception as e:
                logging.error(f"Error processing {sample}: {e}")


def run_wmca(rootpath: str, output_folder: str, sample_split_file: List[str]) -> None:
    """Process WMCA dataset."""
    remove_black_borders = RemoveBlackBorders()

    for file_path in sample_split_file:
        with open(file_path, "r") as file:
            sample_list = [line.strip() for line in file.readlines()]

        for sample in tqdm(sample_list, desc=f"Processing {file_path}"):
            item = sample.split("/")[-1]
            hdf5_file = re.sub(r"_(\d{2})\.jpg$", ".hdf5", item).replace("_", "/", 1)
            hdf5_path = os.path.join(rootpath, hdf5_file)

            if not os.path.exists(hdf5_path):
                logging.warning(f"File not found: {hdf5_path}")
                continue

            try:
                with h5py.File(hdf5_path, "r") as hdf5_file:
                    for name in hdf5_file:
                        frame_number = int(name.split("_")[1])
                        if f"{frame_number:02d}" == item.split(".jpg")[0].split("_")[-1] and isinstance(
                            hdf5_file[name], h5py.Group
                        ):
                            for subname in hdf5_file[name]:
                                dataset = hdf5_file[f"{name}/{subname}"]
                                frame = np.array(dataset)

                                if frame.ndim == 3 and frame.shape[0] == 1:
                                    frame = frame[0]
                                elif frame.ndim == 1:
                                    frame = frame.reshape((1, 1, frame.shape[0]))
                                elif frame.ndim == 2:
                                    frame = frame.reshape((frame.shape[0], frame.shape[1], 1))
                                elif frame.shape == (3, 128, 128):
                                    frame = frame.transpose(1, 2, 0)

                                img = Image.fromarray(frame)
                                target_path = os.path.join(output_folder, sample)
                                process_image(img_path=img, output_path=target_path, remove_black_borders=remove_black_borders)
            except Exception as e:
                logging.error(f"Error processing {hdf5_path}: {e}")


def run_surf(rootpath: str, output_folder: str, sample_split_file: List[str]) -> None:
    """Process SURF dataset."""
    remove_black_borders = RemoveBlackBorders()

    for file_path in sample_split_file:
        with open(file_path, "r") as file:
            sample_list = [line.strip() for line in file.readlines()]

        for sample in tqdm(sample_list, desc=f"Processing {file_path}"):
            item = sample.split("/")[-1]
            parts = item.split("_")

            if "Training" in parts:
                dataset_path = os.path.join(
                    rootpath,
                    parts[0],
                    f"{parts[1]}_{parts[2]}",
                    f"{parts[3]}_{parts[4]}",
                    f"{parts[5]}_{parts[6]}_{parts[7]}" if f"{parts[1]}_{parts[2]}" == "fake_part" else parts[5],
                    parts[-2],
                    parts[-1],
                )
            elif "Val" in parts or "Testing" in parts:
                dataset_path = os.path.join(rootpath, parts[0], parts[1], parts[2])
            else:
                logging.warning(f"Unrecognized path format: {item}")
                continue

            if not os.path.exists(dataset_path):
                logging.warning(f"File not found: {dataset_path}")
                continue

            try:
                target_path = os.path.join(output_folder, sample)
                process_image(img_path=dataset_path, output_path=target_path, remove_black_borders=remove_black_borders)
            except Exception as e:
                logging.error(f"Error processing {dataset_path}: {e}")


if __name__ == "__main__":
    run_cefa(
        rootpath=cfg.cefa_path,
        output_folder=cfg.WCS_frame_path,
        sample_split_file=[
            cfg.WCS_txt_label + "/cefa_fake_test.txt",
            cfg.WCS_txt_label + "/cefa_fake_train.txt",
            cfg.WCS_txt_label + "/cefa_real_test.txt",
            cfg.WCS_txt_label + "/cefa_real_train.txt",
            cfg.WCS_txt_label + "/cefa_real_shot.txt",
            cfg.WCS_txt_label + "/cefa_fake_shot.txt",
        ],
    )

    run_wmca(
        rootpath=cfg.wmca_path,
        output_folder=cfg.WCS_frame_path,
        sample_split_file=[
            cfg.WCS_txt_label + "/wmca_fake_test.txt",
            cfg.WCS_txt_label + "/wmca_fake_train.txt",
            cfg.WCS_txt_label + "/wmca_real_test.txt",
            cfg.WCS_txt_label + "/wmca_real_train.txt",
            cfg.WCS_txt_label + "/wmca_real_shot.txt",
            cfg.WCS_txt_label + "/wmca_fake_shot.txt"
        ],
    )

    run_surf(
        rootpath=cfg.surf_path,
        output_folder=cfg.WCS_frame_path,
        sample_split_file=[
            cfg.WCS_txt_label + '/surf_fake_test.txt',
            cfg.WCS_txt_label + '/surf_fake_train.txt',
            cfg.WCS_txt_label + '/surf_real_test.txt',
            cfg.WCS_txt_label + "/surf_real_train.txt",
            cfg.WCS_txt_label + '/surf_real_shot.txt',
            cfg.WCS_txt_label + "/surf_real_shot.txt",
        ],
    )
