# About
This is a set of tools I've developed for working with crowdsourced annotations of aerial wildlife images from Zooniverse as part of the "Drones for Ducks" project. After processing, these annotations are used to train and validate deep learning detection and classification models to automate the processing of imagery from aerial wildlife surveys. 

# Zooniverse Workflow
These tools assume the following workflow
* Tile full size aerial images to 640 x 640
* Pick images with birds (first manually, later with a deep learning detector) for upload to Zooniverse
* Download Zooniverse CSV with crowdsourced image annotations
* Use the following tools to convert annotations to COCO format, aggregate redundant annotations, calculate agreement metrics per class, and restitch annotations to full size aerial images 

# Contents
* *zooniverse_tools:* module for ingesting raw Zooniverse CSV classification download of a rectangle-based detection + classification task and translate to COCO format. Aggregates redundant annotations from multiple users via DBSCAN for detections and plurality vote for classifications
  * Will also optionally calculate agreement metrics for the aggregated annotations and restitch tiled annotations to full size images
* *tile:* batch tiling of aerial imagery to specified dimensions (usually 640 x 640). Editable header for implementation.
* *process_zooniverse:* implementation script (editable header) to process raw Zooniverse data.
   * Outputs:
     * COCO JSON of raw annotations
     * COCO JSON of aggregated annotations
     * OPTIONAL: COCO JSON of tiled annotations restitched to full size aerial images, sample preview images with annotations drawn on to check restitching worked correctly
