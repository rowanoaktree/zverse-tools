## Generate YOLO-size tiles from a directory of images
## This will generate precise size tiles to avoid image distortion during training
## However, this leads to uneven sizes at image edges, so these are sorted into "good" and "bad" folders

from PIL import Image
import os
import math
from pathlib import Path

originals_dir = Path("D:\\working\\whcr_batches")
tile_size = (640, 640) # Size of the annotated tiles (will almost always be 640, 640 for Zooniverse upload limits))
tiles_dir = Path("D:\\working\\previews2") # Path to a directory where the tiles will be saved

def tile_image(image_path, output_dir, tile_size):
    img = Image.open(image_path)
    width, height = img.size
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    tile_width, tile_height = tile_size
    cols = math.ceil(width / tile_width)
    rows = math.ceil(height / tile_height)

    # Output subfolders
    good_dir = os.path.join(output_dir, "good_shapes")
    bad_dir = os.path.join(output_dir, "bad_shapes")
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)

    for row in range(rows):
        for col in range(cols):
            left = col * tile_width
            upper = row * tile_height
            right = min(left + tile_width, width)
            lower = min(upper + tile_height, height)
            tile = img.crop((left, upper, right, lower))

            tile_filename = f"{base_name}_{col}_{row}.png"

            # Save to the appropriate folder
            if tile.size == (tile_width, tile_height):
                save_path = os.path.join(good_dir, tile_filename)
            else:
                save_path = os.path.join(bad_dir, tile_filename)

            tile.save(save_path)

def batch_tile_images(input_dir, output_dir, tile_size=(640, 640)):
    supported_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_exts):
            image_path = os.path.join(input_dir, filename)
            print(f"Tiling {filename}...")
            tile_image(image_path, output_dir, tile_size)

# IMPLEMENTATION:
batch_tile_images(
    originals_dir,
    tiles_dir,
    tile_size=(640, 640)
)