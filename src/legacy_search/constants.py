"""Data path templates for the legacy object-detection filter.

Paths are relative to the process's working directory, matching how the
rest of the legacy pipeline resolves `data/`.
"""

MAP_KEYFRAMES_CSV_PATH = "data/map-keyframes/{video_name}.csv"
OBJECT_DETECTIONS_JSON_PATH = "data/objects/{video_name}/{index:03d}.json"
