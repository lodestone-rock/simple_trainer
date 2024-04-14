from image_dataloader import download_dataset, ImageDataset

DATASETS_CREDS = "hf_datasets.json"
chunk_name = download_dataset(DATASETS_CREDS, chunks=0)
image = ImageDataset(csv_path="temp/256-polyfur-006542f8-bef5-4fcd-8d25-45940c701437.csv", caption_col="caption", filename_col="filename", image_dir="temp")
image[23]

print()
