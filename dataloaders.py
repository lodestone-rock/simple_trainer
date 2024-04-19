import os
import time
import random
from queue import Queue
from typing import Optional, List
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import subprocess

import requests
import cv2
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
import random
from queue import Queue
from google.cloud import storage
import logging

from tqdm import tqdm


def threading_dataloader(
    dataset,
    batch_size=1,
    num_workers=10,
    collate_fn=None,
    shuffle=False,
    prefetch_factor=4,
    seed=0,
    timeout=None,
):
    """
    A function to load data using multiple threads. This function can be used to speed up the data loading process.

    Parameters:
    dataset (iterable): The dataset to load.
    batch_size (int, optional): The number of samples per batch. Defaults to 1.
    num_workers (int, optional): The number of worker threads to use. Defaults to 10.
    collate_fn (callable, optional): A function to collate samples into a batch. If None, the default collate_fn is used.
    shuffle (bool, optional): Whether to shuffle the dataset before loading. Defaults to False.
    prefetch_factor (int, optional): The number of batches to prefetch. Defaults to 4.
    seed (int, optional): The seed for the random number generator. Defaults to 0.
    timeout (int, optional): The maximum number of seconds to wait for a batch. If None, there is no timeout.

    Yields:
    object: A batch of data.
    """
    # Initialize a random number generator with the given seed
    random.seed(seed)

    # Create a ThreadPoolExecutor with the specified number of workers
    workers = ThreadPoolExecutor(max_workers=num_workers)
    overseer = ThreadPoolExecutor(max_workers=2)

    # Generate batches of indices based on the dataset size and batch size
    num_samples = len(dataset)
    batch_indices = [
        list(range(i, min(i + batch_size, num_samples)))
        for i in range(0, num_samples, batch_size)
    ]
    if shuffle:
        indices = list(range(num_samples))
        random.shuffle(indices)
        batch_indices = [
            indices[i : i + batch_size] for i in range(0, num_samples, batch_size)
        ]

    # Create a queue to store prefetched batches
    prefetch_queue = Queue(maxsize=prefetch_factor * num_workers)

    # Function to prefetch batches of samples
    def batch_to_queue(indices):
        """
        Function to load a batch of data and put it into the prefetch queue.

        Parameters:
        indices (list): The indices of the samples in the batch.
        """
        batch = [dataset[i] for i in indices]
        if collate_fn is not None:
            batch = collate_fn(batch)
        prefetch_queue.put(
            batch
        )  # 1. if you want to ensure order use return instead of queue here

    # Submit the prefetch tasks to the worker threads
    def overseer_thread():
        for indices in batch_indices:
            workers.submit(
                batch_to_queue, indices
            )  # 2. then store the future and re itterate it here and get the value

    def monitor_queue():
        while True:
            time.sleep(10)
            print("dataloader queue:", prefetch_queue.qsize())
            print("worker queue:", workers._work_queue.qsize())
            if workers._work_queue.qsize() + prefetch_queue.qsize() == 0:
                print("all task is done, shutting down threads")
                overseer.shutdown()
                workers.shutdown()

    # just in case worker submit loop is too slow due to a ton of loops, fork it to another thread so the main thread can continue
    overseer.submit(overseer_thread)
    overseer.submit(monitor_queue)

    # Yield the prefetched batches
    for _ in range(len(batch_indices)):
        yield prefetch_queue.get(timeout=timeout)


def list_bucket_contents(bucket_name, service_account_json):
    # Create a client using the provided service account JSON file
    client = storage.Client.from_service_account_json(service_account_json)

    # Access the specified bucket
    bucket = client.get_bucket(bucket_name)

    # List the contents of the bucket
    blobs = bucket.list_blobs()
    blobs_name = []
    for blob in blobs:
        blobs_name.append((blob.name, blob.size, blob.md5_hash))
    return blobs_name


def download_blob(
    bucket_name, service_account_json, source_blob_name, destination_file_name
):
    """Downloads a blob from the bucket."""
    client = storage.Client.from_service_account_json(service_account_json)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


def sample_frames_from_video(
    video_path, output_folder, num_frames_to_sample, start_offset=0, end_offset=None
):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        return

    # Get total number of frames in the video
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Calculate start and end frame indices based on time offsets
    start_frame_index = int(start_offset * fps)
    if end_offset is not None:
        end_frame_index = min(int(end_offset * fps), total_frames)
    else:
        end_frame_index = total_frames

    # Calculate frame sampling interval
    sampling_interval = max(
        (end_frame_index - start_frame_index) // num_frames_to_sample, 1
    )

    # Initialize sampled frame count and current frame index
    sampled_frame_count = 0
    current_frame_index = start_frame_index

    # Read until video is completed or end frame is reached
    while current_frame_index < end_frame_index:
        # Set the frame index
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)

        # Read the frame
        ret, frame = video_capture.read()

        # Break the loop if no frame is read
        if not ret:
            break

        # Save the frame as an image
        frame_path = os.path.join(
            output_folder, f"sampled_frame_{sampled_frame_count}.jpg"
        )
        cv2.imwrite(frame_path, frame)

        # Increment sampled frame count
        sampled_frame_count += 1

        # Check if all sampled frames are captured
        if sampled_frame_count == num_frames_to_sample:
            break

        # Move to the next frame based on sampling interval
        current_frame_index += sampling_interval

    # Release the video capture object
    video_capture.release()

    print(f"{num_frames_to_sample} frames sampled from the video.")


class E6Downloader:
    def __init__(
        self,
        e6_csv_dump: str,
        extension: List,
        service_account_json: str,
        bucket_name: str,
        temp_folder: str = "temp",
        delete: bool = True,
    ):
        # Configure the logging
        logging.basicConfig(
            filename="error.log",
            level=logging.ERROR,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # data filtering
        self.data = pd.read_csv(e6_csv_dump)
        self.data = self.data[self.data["file_ext"].isin(extension)][::-1]
        self.folder = temp_folder
        self.delete = delete
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        # gcs
        self.service_account_json = service_account_json
        self.bucket_name = bucket_name

    def upload_file_to_gcs(self, local_file_path, destination_blob_name):
        # Create a client using the service account JSON file
        client = storage.Client.from_service_account_json(self.service_account_json)

        # Get the bucket
        bucket = client.get_bucket(self.bucket_name)

        # Upload the file to the bucket
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {}
        try:
            # download via wget
            post = self.data.iloc[idx]

            url = f"https://static1.e621.net/data/{post.md5[0:2]}/{post.md5[2:4]}/{post.md5}.{post.file_ext}"
            command = f"wget {url} -O {self.folder}/{post.md5}.{post.file_ext}"

            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                self.upload_file_to_gcs(
                    f"{self.folder}/{post.md5}.{post.file_ext}",
                    f"{post.md5}.{post.file_ext}",
                )
            # delete the temp files
            if (
                os.path.exists(f"{self.folder}/{post.md5}.{post.file_ext}")
                and self.delete
            ):
                os.remove(f"{self.folder}/{post.md5}.{post.file_ext}")
            return 0

        except Exception as e:
            self.logger.error("An error occurred: %s", str(e))
            return e
