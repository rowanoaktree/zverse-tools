## Module for processing raw Zooniverse CSV downloads into COCO format
## Includes functions for aggregating redundant annotations, calculating agreement metrics between observers, and restitching annotations on tiles to full size images

## === Imports ===

import json
import pandas as pd
import numpy as np
from shapely.geometry import box
import ast
import os
import re
from PIL import Image, ImageDraw
from collections import defaultdict
from copy import deepcopy
import random
from sklearn.cluster import DBSCAN

## === Functions ===

## Convert raw annotations from Zooniverse CSV download to COCO format
def zoo2coco(input_csv, output_json):
    print("[Step 1] Converting Zooniverse CSV to COCO format...")

    try:
        zooniverse = pd.read_csv(input_csv)
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")
        return

    images = {}
    annos = []
    categories = {}
    labelers = {}
    bbox_dimensions = []

    # First pass: collect valid bounding boxes to estimate median sizes
    for idx, row in zooniverse.iterrows():
        labeler = row.get("user_name", f"user_{idx}")
        if labeler not in labelers:
            labelers[labeler] = len(labelers) + 1
        labeler_id = labelers[labeler]

        try:
            imgrow = json.loads(row["subject_data"])
        except Exception as e:
            print(f"[WARN] Could not parse subject_data at row {idx}: {e}")
            continue

        try:
            image_name = next(iter(imgrow.values()))["Filename"]
        except Exception:
            print(f"[WARN] Could not extract image name at row {idx}")
            continue

        if image_name not in images:
            images[image_name] = len(images) + 1
        image_id = images[image_name]

        try:
            annotations = json.loads(row["annotations"])
        except Exception as e:
            print(f"[WARN] Could not parse annotations at row {idx}: {e}")
            continue

        for task in annotations:
            if task.get("task") != "T1":
                continue

            for ann in task.get("value", []):
                w, h = ann.get("width"), ann.get("height")
                if w is not None and h is not None:
                    bbox_dimensions.append((w, h))

    if not bbox_dimensions:
        print("[ERROR] No valid bounding boxes found.")
        return

    bbox_array = np.array(bbox_dimensions)
    median_width = np.median(bbox_array[:, 0])
    median_height = np.median(bbox_array[:, 1])
    print(f"[INFO] Median Width: {median_width:.2f}, Median Height: {median_height:.2f}")

    # Second pass: build annotations using medians if needed
    for idx, row in zooniverse.iterrows():
        labeler = row.get("user_name", f"user_{idx}")
        labeler_id = labelers.get(labeler, len(labelers) + 1)

        try:
            imgrow = json.loads(row["subject_data"])
            image_name = next(iter(imgrow.values()))["Filename"]
        except Exception:
            continue

        image_id = images[image_name]

        try:
            annotations = json.loads(row["annotations"])
        except:
            continue

        for task in annotations:
            if task.get("task") != "T1":
                continue

            for ann in task.get("value", []):
                try:
                    x = ann["x"]
                    y = ann["y"]
                    w = ann.get("width", median_width)
                    h = ann.get("height", median_height)
                    label = ann["tool_label"]
                except KeyError:
                    continue

                bbox = [x, y, w, h]
                area = w * h

                if label not in categories:
                    categories[label] = len(categories) + 1
                category_id = categories[label]

                annotation = {
                    "annotation_id": len(annos) + 1,
                    "bbox": bbox,
                    "area": area,
                    "category_id": category_id,
                    "category": label,
                    "image_id": image_id,
                    "filename": image_name,
                    "labeler_id": labeler_id
                }
                annos.append(annotation)

    # Prepare basic COCO-like structure
    coco_output = {
        "images": [{"id": v, "file_name": k} for k, v in images.items()],
        "annotations": annos,
        "categories": [{"id": v, "name": k} for k, v in categories.items()],
        "labelers": [{"id": v, "name": k} for k, v in labelers.items()]
    }

    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(coco_output, f, indent=2, ensure_ascii=False)
        print(f"[Step 1 Complete] COCO file saved to: {output_json}")
    except Exception as e:
        print(f"[ERROR] Could not save COCO file: {e}")

## -- Set of functions to aggregate redundant annotations and calculate agreement between observers --

## Run agreement metrics on the clustered annotations
def ann_agreement(df, refined, category_map):
    print("[Step 2A] Running agreement metrics...")

    # Merge raw annotations with refined consensus per cluster
    merged = df.merge(refined, on=["image_id", "cluster_id"], how="left", suffixes=("_orig", "_ref"))

    # Clean columns and rename for clarity
    merged = merged.rename(columns={
        "bbox_orig": "bbox",
        "category_id_orig": "cat_id_orig",
        "category_id_ref": "cat_id_refined"
    })

    # Label name columns
    merged["cat_orig"] = merged["cat_id_orig"].map(category_map)
    merged["cat_refined"] = merged["cat_id_refined"].map(category_map)

    # Agreement indicator
    merged["agree"] = (merged["cat_orig"] == merged["cat_refined"]).astype(int)

    # Agreement summary
    overall_agree_mean = merged["agree"].mean()
    overall_agree_std = merged["agree"].std()
    by_class = merged.groupby("cat_refined")["agree"].agg(["mean", "std"])

    print(f"[Metrics] Overall label agreement: {overall_agree_mean:.3f} ± {overall_agree_std:.3f}")
    print("[Metrics] Label agreement by class:")
    print(by_class)

    # IOU calculation
    def eval_bbox(bbox):
        if isinstance(bbox, str):
            try:
                bbox = ast.literal_eval(bbox)
            except Exception:
                return np.array([np.nan]*4)
        return np.array(bbox, dtype=float)

    def calculate_iou(row):
        b1 = eval_bbox(row["bbox"])
        b2 = eval_bbox(row["bbox_ref"])
        if np.any(np.isnan(b1)) or np.any(np.isnan(b2)):
            return np.nan
        box1 = box(b1[0], b1[1], b1[0]+b1[2], b1[1]+b1[3])
        box2 = box(b2[0], b2[1], b2[0]+b2[2], b2[1]+b2[3])
        return box1.intersection(box2).area / box1.union(box2).area

    merged["IOU"] = merged.apply(calculate_iou, axis=1)

    iou_mean = merged["IOU"].mean()
    iou_by_class = merged.groupby("cat_refined")["IOU"].mean()

    print(f"[Metrics] Overall IOU: {iou_mean:.3f}")
    print("[Metrics] IOU by class:")
    print(iou_by_class)

    # Pielou's evenness index
    pielou_results = []
    grouped = merged.groupby(["image_id", "cluster_id"])
    for (img_id, clus_id), group in grouped:
        cat_counts = group["cat_id_orig"].value_counts()
        if len(cat_counts) <= 1:
            evenness = 0
        else:
            rel_abund = cat_counts / cat_counts.sum()
            evenness = -np.sum(rel_abund * np.log(rel_abund)) / np.log(len(rel_abund))
        pielou_results.append({
            "image_id": img_id,
            "cluster_id": clus_id,
            "pielou_index": evenness,
            "cat_refined": group["cat_refined"].iloc[0]
        })

    pielou_df = pd.DataFrame(pielou_results)
    print(f"[Metrics] Pielou Index: mean={pielou_df['pielou_index'].mean():.3f}, std={pielou_df['pielou_index'].std():.3f}")
    print("[Metrics] Pielou Index by class:")
    print(pielou_df.groupby("cat_refined")["pielou_index"].agg(["mean", "std"]))

    print("[Step 2A] Agreement metrics complete.")

## DBSCAN implementation for clustering annotations using annotation center coordinates
def group_DBSCAN(df):
    x = df[["c_x", "c_y"]].to_numpy()
    cluster = DBSCAN(eps=15, min_samples=5).fit(x)
    labels = cluster.labels_
    df["cluster_id"] = labels
    return labels

## Group annotations by image tile, apply DBSCAN clustering for detections, and majority vote for classifications
## Save aggregated annotations to JSON
def agg_anns(input_json, output_json, compute_metrics=True):
    print("[Step 2] Aggregating redundant annotations...")

    # Load annotations from JSON
    try:
        with open(input_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data["annotations"])
    except Exception as e:
        print(f"[ERROR] Could not load JSON or parse annotations: {e}")
        return

    # Extract bounding box center coordinates and components
    bbox_list = df["bbox"].tolist()
    if not all(isinstance(b, (list, tuple)) and len(b) == 4 for b in bbox_list):
        print("[ERROR] Invalid bounding box format.")
        return

    df["x"] = [b[0] for b in bbox_list]
    df["y"] = [b[1] for b in bbox_list]
    df["w"] = [b[2] for b in bbox_list]
    df["h"] = [b[3] for b in bbox_list]
    df["c_x"] = df["x"] + df["w"] / 2
    df["c_y"] = df["y"] + df["h"] / 2

    # Apply DBSCAN per image and attach cluster labels
    df["cluster_id"] = (
        df.groupby("filename", group_keys=False)
        .apply(lambda g: pd.Series(group_DBSCAN(g), index=g.index))
    )

    # Get dynamic category mapping
    category_map = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    if not category_map and "category" in df.columns:
        category_map = df.set_index("category_id")["category"].dropna().to_dict()

    def safe_mode(series):
        modes = pd.Series(series).mode()
        return modes.iloc[0] if not modes.empty else np.nan

    # Create image ID map
    unique_filenames = df["filename"].unique()
    image_id_map = {fname: idx + 1 for idx, fname in enumerate(unique_filenames)}
    df["image_id"] = df["filename"].map(image_id_map)

    ## Aggregate median bbox and majority vote category per cluster
    refined = df.groupby(["image_id", "cluster_id"]).agg({
        "x": "median",
        "y": "median",
        "w": "median",
        "h": "median",
        "category_id": safe_mode
    }).reset_index()

    # Map back to filenames
    refined["filename"] = refined["image_id"].map({v: k for k, v in image_id_map.items()})

    # Drop rows with no valid category ID
    refined = refined[refined["category_id"].notnull()]
    refined["category_id"] = refined["category_id"].astype(int)

    # Drop noise clusters
    before_noise = len(refined)
    refined = refined[refined["cluster_id"] != -1]
    after_noise = len(refined)

    # Compose bbox list
    refined["bbox"] = refined[["x", "y", "w", "h"]].values.tolist()
    refined["category"] = refined["category_id"].map(category_map)

    # For writing to JSON
    final_annotations = refined.drop(columns=["x", "y", "w", "h", "cluster_id", "filename"])

    # Summary stats
    removed_due_to_label = df["cluster_id"].nunique() - len(refined)
    noise_removed = before_noise - after_noise
    print(f"[INFO] Removed {removed_due_to_label} clusters due to no agreement on class ID")
    print(f"[INFO] Removed {noise_removed} noise clusters")

    # Optional: Run agreement metrics
    if compute_metrics:
        ann_agreement(df, refined, category_map)

    # Prepare output
    image_list = [{"id": image_id_map[fname], "file_name": fname} for fname in unique_filenames]
    category_items = [{"id": cid, "name": name} for cid, name in category_map.items()]
    aggregated_output = {
        "images": image_list,
        "annotations": final_annotations.to_dict(orient="records"),
        "categories": category_items
    }

    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(aggregated_output, f, indent=2, ensure_ascii=False)
        print(f"[Step 2] Aggregated annotations saved to: {output_json}")
    except Exception as e:
        print(f"[ERROR] Could not write output JSON: {e}")

## -- Set of functions to restitch annotations on tiles to original --

## Helper functions to parse tile names and load original image sizes
def parse_tile_name(filename):
    match = re.match(r'(.+)_([0-9]+)_([0-9]+)\.png$', filename)
    if match:
        base_name = match.group(1)
        col = int(match.group(2))
        row = int(match.group(3))
        return base_name, col, row
    return None, None, None

def load_original_image_sizes(originals_dir, supported_exts=('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
    size_lookup = {}
    for root, _, files in os.walk(originals_dir):
        for fname in files:
            if fname.lower().endswith(supported_exts):
                base_name = os.path.splitext(fname)[0]
                img_path = os.path.join(root, fname)
                try:
                    with Image.open(img_path) as im:
                        size_lookup[base_name] = im.size  # (width, height)
                except Exception as e:
                    print(f"Failed to load image {fname}: {e}")
    return size_lookup

## Draw annotations on images for preview to ensure stitching worked correctly
def draw_annotations_preview(full_images, annotations, output_dir, originals_dir, max_samples=None):
    os.makedirs(output_dir, exist_ok=True)
    ann_by_image = defaultdict(list)
    for ann in annotations:
        ann_by_image[ann['image_id']].append(ann)

    image_list = list(full_images)
    if max_samples is not None and max_samples < len(image_list):
        image_list = random.sample(image_list, max_samples)
        print(f"🔍 Generating previews for a random sample of {max_samples} images...")

    for image_entry in image_list:
        image_id = image_entry['id']
        filename = image_entry['file_name']
        base_name = os.path.splitext(filename)[0]

        original_path = None
        for root, _, files in os.walk(originals_dir):
            for fname in files:
                if fname.startswith(base_name):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
                        original_path = os.path.join(root, fname)
                        break
            if original_path:
                break

        if not original_path:
            print(f"Original image for preview not found: {base_name}")
            continue

        try:
            with Image.open(original_path) as img:
                draw = ImageDraw.Draw(img)
                for ann in ann_by_image[image_id]:
                    x, y, w, h = ann['bbox']
                    draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
                preview_path = os.path.join(output_dir, base_name + "_preview.jpg")
                img.convert("RGB").save(preview_path, "JPEG", quality=90)
                print(f"Saved preview: {preview_path}")
        except Exception as e:
            print(f"Failed to draw preview for {filename}: {e}")

## Main function to stitch annotations from tiles to full images
def stitch_anns(coco_tile_json, tile_size, originals_dir, output_json_path, preview_dir=None):
    with open(coco_tile_json, 'r') as f:
        data = json.load(f)

    tile_width, tile_height = tile_size
    image_sizes = load_original_image_sizes(originals_dir)

    tile_image_dict = {img['id']: img for img in data['images']}
    full_images = {}
    image_id_map = {}
    new_annotations = []
    next_image_id = 1
    next_ann_id = 1

    for ann in data['annotations']:
        tile_img = tile_image_dict.get(ann['image_id'])
        if tile_img is None:
            print(f"Skipping annotation {ann['id']}: image_id {ann['image_id']} not found.")
            continue

        tile_filename = tile_img.get('file_name')
        base_name, col, row = parse_tile_name(tile_filename)
        if base_name is None or col is None or row is None:
            print(f"Skipping {tile_filename}: cannot parse tile name.")
            continue
        if base_name not in image_sizes:
            print(f"Skipping {tile_filename}: original image {base_name} not found.")
            continue

        x_offset = col * tile_width
        y_offset = row * tile_height

        if base_name not in full_images:
            width, height = image_sizes[base_name]
            full_images[base_name] = {
                "id": next_image_id,
                "file_name": base_name + ".jpg",  # change extension if needed
                "width": width,
                "height": height,
            }
            image_id_map[base_name] = next_image_id
            next_image_id += 1

        x, y, w, h = ann['bbox']
        new_bbox = [x + x_offset, y + y_offset, w, h]

        new_ann = deepcopy(ann)
        new_ann['id'] = next_ann_id
        new_ann['image_id'] = image_id_map[base_name]
        new_ann['bbox'] = new_bbox
        new_annotations.append(new_ann)
        next_ann_id += 1

    final_data = {
        "images": list(full_images.values()),
        "annotations": new_annotations,
        "categories": data['categories'],
    }

    with open(output_json_path, 'w') as f:
        json.dump(final_data, f, indent=2)
    print(f"\n✅ Stitched annotations saved to {output_json_path}")

    if preview_dir:
        draw_annotations_preview(
        full_images=final_data['images'],
        annotations=final_data['annotations'],
        output_dir=preview_dir,
        originals_dir=originals_dir,
        max_samples=20  # set to None or omit to preview all
    )