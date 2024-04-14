
import os
from huggingface_hub import HfFileSystem, hf_hub_url
import subprocess
import cv2
import json
import bisect
import pandas as pd


def scale(image, scale_factor):
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image


def download_dataset(json_creds:str, download_dir:str="temp", image_dir:str="temp", chunks:int=0):
    # create temp folder to store zip file
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Open the JSON file
    with open(json_creds, 'r') as file:
        # Load the JSON data from the file
        hf_creds = json.load(file)
    
    # Initialize the Hugging Face FileSystem with the provided token
    fs = HfFileSystem(token=hf_creds['token'])

    # Get a list of all files inside the specified directory
    file_paths = fs.ls(f"datasets/{hf_creds['repo']}/chunks", detail=False)
    # assume zip url is sorted 
    zip_file = [file for file in file_paths if file.endswith('.zip')][chunks]
    zip_url = f"https://huggingface.co/datasets/{hf_creds['repo']}/resolve/main/chunks/{zip_file.split('/')[-1]}"
    csv_url = f"https://huggingface.co/datasets/{hf_creds['repo']}/resolve/main/chunks/{zip_file.split('/')[-1].split('.')[0]}.csv"
    # download the zip
    token = hf_creds['token']

    # Construct the wget command with the Authorization header
    wget_zip = f'wget --header="Authorization: Bearer {token}" {zip_url} -P {download_dir}'

    # Execute the command using subprocess
    process = subprocess.Popen(wget_zip, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    wget_csv = f'wget --header="Authorization: Bearer {token}" {csv_url} -P {download_dir}'

    # Execute the command using subprocess
    process = subprocess.Popen(wget_csv, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    # unzip 
    subprocess.run(["7z", "x", f"{download_dir}/{zip_file.split('/')[-1]}", f"-o{image_dir}", "-Y"])
    os.remove(f"{download_dir}/{zip_file.split('/')[-1]}")
    return zip_file.split('/')[-1].split('.')[0] # return chunk name for easier navigation later


class ImageDataset():
    def __init__(self, csv_path:str, caption_col:str, filename_col:str, image_dir="temp_image",  round_to=64):
        self.IMAGE_DIR = image_dir
        self.round_to = round_to
        self.file_list = os.listdir(image_dir)
        # read from preformated csv 
        self.csv_path = csv_path
        self.caption_col = caption_col
        self.filename_col = filename_col

        # lut is hardcoded for now
        # can be computed using 1/x fn where x < max_square_res but eh lazy
        self.RES_LUT = [512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1344, 1472, 1600, 1792, 2048]
        self.ar_lut = [
            [512, 2048],
            [576, 1792],
            [640, 1600],
            [704, 1472],
            [768, 1344],
            [832, 1216],
            [896, 1152],
            [960, 1088],
            [1024, 1024]
        ]
        self.ar_lut_frac = [height/width for (height, width) in self.ar_lut]

        self.captions = self._get_dict_captions()

    
    def _get_dict_captions(self):
        # create caption dictionary to be used in __getitem__
        df = pd.read_csv(self.csv_path)
        return dict(zip(df[self.filename_col], df[self.caption_col]))


    def __len__(self):
        return len(os.listdir(self.IMAGE_DIR))

    def __getitem__(self, idx):
        sample = {}
        try:
            img_name = os.path.join(self.IMAGE_DIR, self.file_list[idx])
            caption = self.captions[img_name.split("/")[-1]]
            image = cv2.imread(img_name)
            if image.shape[-1] != 3:
                raise "image has more than 3 channel"
            
            # Calculate the dimensions for center crop
            height, width = image.shape[:2]
            # grab the minimum axis as the anchor and crop the max axis

            if width > height:
                image_ar = height/width
                ar_index = bisect.bisect_right(self.ar_lut_frac, image_ar)
                bucket_res = self.ar_lut[ar_index]
                ar_rescale = bucket_res[0] / height  # rescale to match width
                crop_max_axis = bucket_res[1]

            else:
                image_ar = width/height
                ar_index = bisect.bisect_right(self.ar_lut_frac, image_ar)
                bucket_res = self.ar_lut[ar_index]
                ar_rescale = bucket_res[0] / width  # rescale to match width
                crop_max_axis = bucket_res[1]
         
            image = scale(image, ar_rescale)
            # new height and width
            height, width = image.shape[:2]
            
            if width < height:
                top = (height - crop_max_axis) // 2
                bottom = top + crop_max_axis
                image = image[top:bottom, :]

            else:
                left = (width - crop_max_axis) // 2
                right = left + crop_max_axis
                image = image[:, left:right]
            


            sample["image"] = image
            sample["caption"] = caption

        except Exception as e:
            print(e)
        
        return sample
