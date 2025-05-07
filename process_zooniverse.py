## Process raw Zooniverse CSV annotations
## Steps:
## 1. Convert raw Zooniverse CSV download to COCO JSON format
## 2. Aggregate redundant annotations from multiple users using DBSCAN and plurality vote

## OPTIONAL STEPS:
## 2a. Calculate agreement metrics in the aggregated annotations 
##  (by default this is turned on, set 'compute_metrics' to False in 'agg_anns' to skip)
## 3. Check number of annotations and image tiles in the aggregated JSON file
## 4. Restitch annotations on image tiles to full size images
## 5. Check number of images and annotations in the stitched JSON file

from pathlib import Path
from zooniverse_tools import zoo2coco, agg_anns, ann_agreement, stitch_anns
import json
## Paths for inputs and outputs
raw_csv = Path("C:\\Users\\rowanconverse\\OneDrive - University of New Mexico\\Projects\\WHCR\\cranes-and-crane-look-alikes-classifications.csv") # Path to the raw CSV file downloaded from Zooniverse
raw_json = Path("C:\\Users\\rowanconverse\\OneDrive - University of New Mexico\\Projects\\WHCR\\test1\\raw_coco.json") # Path to save the raw annotations in COCO format
aggregated_json = Path("C:\\Users\\rowanconverse\\OneDrive - University of New Mexico\\Projects\\WHCR\\test1\\aggregated_coco.json") # Path to save the aggregated annotations

zoo2coco(raw_csv, raw_json)
agg_anns(raw_json, aggregated_json)

## OPTIONAL: Check number of annotations and images in the aggregated JSON file
try:
    with open(aggregated_json, 'r') as file:
        file_data = json.load(file)
        anns = file_data.get('annotations')
        imgs = file_data.get('images')
        if anns and imgs:
            print(f"Number of images: {len(imgs)}")
            print(f"Number of annotations: {len(anns)}")
        else:
            print("No data found in file")
except FileNotFoundError:
    print("File not found.")
except json.JSONDecodeError:
    print("Invalid JSON format in file.")

## Optional: restitch annotations on image tiles to full size images
#stitched_json = Path("C:\\Users\\rowanconverse\\OneDrive - University of New Mexico\\Projects\\WHCR\\test1\\restitched_coco.json") # Path to save the annotations to full size images
#originals_dir = Path("D:\\working\\whcr_batches") # Path to the original images to derive image dimensions
#tile_size = 640, 640 # Size of the annotated tiles (will almost always be 640, 640)
## Path to a directory where a sample of 20 annotated images will be generated to check that the stitching worked correctly
#previews_dir = Path("D:\\working\\previews2")
#stitch_anns(aggregated_json, tile_size, originals_dir, stitched_json, previews_dir)

## Optional optional: check number of images and annotations in the stitched JSON file
#try:
#    with open(stitched_json, 'r') as file:
#        file_data = json.load(file)
#        anns = file_data.get('annotations')
#        imgs = file_data.get('images')
#        if anns and imgs:
#            print(f"Number of images: {len(imgs)}")
#            print(f"Number of annotations: {len(anns)}")
#        else:
#            print("No data found in file")
#except FileNotFoundError:
#    print("File not found.")
#except json.JSONDecodeError:
#    print("Invalid JSON format in file.")