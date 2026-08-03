from sendRequests import search_queries
from video_clustering import cluster_videos
from itertools import count
from video_clustering_schema import ClusteredCandidate
import json
from searcher.TemporalSearcher import TemporalSearcher
from searcher.AmbiguousSearcher import AmbiguousSearcher
import pandas as pd
from json import JSONDecodeError

# from pathlib import Path
# from pydantic import TypeAdapter

def check_object_satisfaction(clustered_videos, object_name_list, objectThreshold):
    required_object_names = set(object_name_list or [])
    for video in clustered_videos:
        video_name = video.video_name[:-4]  # Remove .mp4 extension
        map_frame_id_file = pd.read_csv(f"data/map-keyframes/{video_name}.csv")
        for result in video.results:
            matching_rows = map_frame_id_file[map_frame_id_file['frame_idx'] == result.frame_index]
            if matching_rows.empty:
                result.satisfiedObjects = False
                continue

            index = matching_rows.index[0]
            object_file_path = f"data/objects/{video_name}/{index:03d}.json"

            try:
                with open(object_file_path, "r", encoding="utf-8") as f:
                    object_data = json.load(f)
            except (FileNotFoundError, JSONDecodeError):
                result.satisfiedObjects = False
                continue

            detection_class_names = object_data.get("detection_class_names", [])
            detection_scores = object_data.get("detection_scores", [])
            detected_object_names = {
                detection_class_name
                for detection_class_name, detection_score in zip(detection_class_names, detection_scores)
                if float(detection_score) >= objectThreshold
            }

            result.satisfiedObjects = required_object_names.issubset(detected_object_names)

def temporal_search(query: list[str], top_k_tuple: int, top_k_each_query: int, gamma: float, 
                    searcher_type: str = "TemporalSearcher", objectFilterMode: bool = False, object_name_list: list[str] = None, 
                    objectThreshold: float = 0.5) -> list[tuple[float, list[ClusteredCandidate]]]:
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
    if searcher_type == "AmbiguousSearcher":
        for video in clustered_videos:
            video_name = video.video_name
            results = video.results
            ambiguous_searcher = AmbiguousSearcher(number_of_queries, results, top_k_tuple, query_results, gamma, video_name, counter, objectFilterMode)
            ambiguous_searcher.start_from_last_element()
    else:     
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
                                                , list_prev_indices, list_endable, gamma, video_name, counter, objectFilterMode)
            temporal_searcher.start_from_last_query()
    return [
        {"score": score, "video_name": video_name, "tuple": [candidate.model_dump() for candidate in tuple_]}
        for score, _, video_name, tuple_ in sorted(query_results, key=lambda x: x[0], reverse=True)
    ]

if __name__ == "__main__":
    query_results = temporal_search(["cat", "dog", "bird", "fish"], 100, 100, 0.05, "TemporalSearcher"
                                    , objectFilterMode = True, object_name_list = ["cat", "dog"], objectThreshold = 0.5)
    ### export the result to a json file
    with open("data/temporal_search_results_temporal.json", "w", encoding="utf-8") as f:
        json.dump(query_results, f, ensure_ascii=False, indent=4)