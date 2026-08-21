from .client import search_queries
from .utils import cluster_videos
from itertools import count
import csv
import json
from .searchers.temporal import TemporalSearcher
from .searchers.ambiguous import AmbiguousSearcher, TraversalBudget
from json import JSONDecodeError

from .constants import MAP_KEYFRAMES_CSV_PATH, OBJECT_DETECTIONS_JSON_PATH


def _load_frame_index_map(video_name: str) -> dict[int, int]:
    """frame_idx -> n (the CSV's own 1-indexed row/keyframe number, the key
    the shipped per-frame detection JSON files are named after). First
    occurrence wins on a duplicate frame_idx, matching the pandas
    `.iloc[0]`-on-first-match behavior this replaced."""

    path = MAP_KEYFRAMES_CSV_PATH.format(video_name=video_name)
    mapping: dict[int, int] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            frame_idx = int(row["frame_idx"])
            if frame_idx not in mapping:
                mapping[frame_idx] = int(row["n"])
    return mapping


def check_object_satisfaction(clustered_videos, object_name_list, objectThreshold):
    required_object_names = {name.casefold() for name in (object_name_list or [])}
    for video in clustered_videos:
        video_name = video.video_name[:-4]  # Remove .mp4 extension
        frame_index_map = _load_frame_index_map(video_name)
        for result in video.results:
            index = frame_index_map.get(result.frame_index)
            if index is None:
                result.satisfiedObjects = False
                continue

            object_file_path = OBJECT_DETECTIONS_JSON_PATH.format(video_name=video_name, index=index)

            try:
                with open(object_file_path, "r", encoding="utf-8") as f:
                    object_data = json.load(f)
            except (FileNotFoundError, JSONDecodeError):
                result.satisfiedObjects = False
                continue

            # detection_class_names holds Open Images machine IDs (e.g.
            # "/m/01jfsr"); detection_class_entities holds the human-readable
            # labels (e.g. "Lantern") object_name_list is actually written
            # in - matching against the machine IDs meant this filter could
            # never match anything a real caller would type.
            detection_class_entities = object_data.get("detection_class_entities", [])
            detection_scores = object_data.get("detection_scores", [])
            detected_object_names = {
                detection_class_entity.casefold()
                for detection_class_entity, detection_score in zip(detection_class_entities, detection_scores)
                if float(detection_score) >= objectThreshold
            }

            result.satisfiedObjects = required_object_names.issubset(detected_object_names)


def temporal_search(query: list[str], top_k_tuple: int, top_k_each_query: int, gamma: float,
                    searcher_type: str = "TemporalSearcher", objectFilterMode: bool = False, object_name_list: list[str] = None,
                    objectThreshold: float = 0.5
                    ) -> tuple[list[dict], bool]:
    number_of_queries = len(query)
    clustered_videos = cluster_videos(search_queries(query, top_k_each_query))
    if objectFilterMode:
        check_object_satisfaction(clustered_videos, object_name_list, objectThreshold)
    ####
    # sample_path = Path(__file__).with_name("sample_test_data.json")
    # clustered_videos = TypeAdapter(list[Videos]).validate_python(json.loads(sample_path.read_text(encoding="utf-8")))
    # for video in clustered_videos:
    #     video.results.sort(key=lambda x: (x.frame_index, x.query_id))
    ####
    counter = count()
    query_results = []
    search_truncated = False
    if searcher_type == "AmbiguousSearcher":
        # One budget shared across every video in this request, not one
        # fresh MAX_TRAVERSAL_NODES/MAX_TRAVERSAL_SECONDS per video - a
        # per-video reset let total traversal time multiply with the number
        # of candidate-heavy videos (measured 200-254s across a handful of
        # videos even at a realistic 4-5 queries, once top_k_each_query was
        # raised toward its schema-allowed max). Sharing one budget bounds
        # the whole request to roughly one video's worth of allowance.
        budget = TraversalBudget(max_nodes=AmbiguousSearcher.MAX_TRAVERSAL_NODES, max_seconds=AmbiguousSearcher.MAX_TRAVERSAL_SECONDS)
        for video in clustered_videos:
            video_name = video.video_name
            results = video.results
            ambiguous_searcher = AmbiguousSearcher(number_of_queries, results, top_k_tuple, query_results, gamma, video_name, counter, objectFilterMode, budget=budget)
            search_truncated = ambiguous_searcher.start_from_last_element() or search_truncated
    else:
        budget = TraversalBudget(max_nodes=TemporalSearcher.MAX_TRAVERSAL_NODES, max_seconds=TemporalSearcher.MAX_TRAVERSAL_SECONDS)
        for video in clustered_videos:
            video_name = video.video_name
            results = video.results
            ### preparing
            list_indices = []
            list_prev_indices = []
            list_nearest_indices = []
            list_endable = [0] * len(results)
            for _ in range(number_of_queries):
                list_indices.append([])
                list_nearest_indices.append(-1)
            for id, result in enumerate(results):
                if result.query_id is None or result.query_id >= number_of_queries:
                    list_prev_indices.append(-1)
                    continue
                list_indices[result.query_id].append(id)
                if result.query_id > 0:
                    list_prev_indices.append(list_nearest_indices[result.query_id - 1])
                else :
                    list_prev_indices.append(-1)
                list_endable[id] = result.query_id == 0 or list_prev_indices[id] != -1
                if list_endable[id]:
                    list_nearest_indices[result.query_id] = len(list_indices[result.query_id]) - 1
            # print(f"video_name: {video_name}, number_of_results: {len(results)}, number_of_queries: {number_of_queries}, top_k_tuple: {top_k_tuple}, top_k_each_query: {top_k_each_query}, gamma: {gamma}")
            # print(f"list_indices: {list_indices}")
            # print(f"list_prev_indices: {list_prev_indices}")
            # print(f"list_endable: {list_endable}")
            temporal_searcher = TemporalSearcher(number_of_queries, results, top_k_tuple, query_results, list_indices
                                                , list_prev_indices, list_endable, gamma, video_name, counter, objectFilterMode, budget=budget)
            search_truncated = temporal_searcher.start_from_last_query() or search_truncated

    results = [
        {
            "score": score,
            "video_name": video_name,
            "tuple": [candidate.model_dump() for candidate in tuple_],
        }
        for score, _, video_name, tuple_ in sorted(query_results, key=lambda x: x[0], reverse=True)
    ]
    return results, search_truncated

if __name__ == "__main__":
    query_results, _truncated = temporal_search(["cat", "dog", "bird", "fish"], 100, 100, 0.05, "TemporalSearcher"
                                    , objectFilterMode = True, object_name_list = ["cat", "dog"], objectThreshold = 0.5)
    ### export the result to a json file
    with open("data/temporal_search_results_temporal.json", "w", encoding="utf-8") as f:
        json.dump(query_results, f, ensure_ascii=False, indent=4)