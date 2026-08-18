from .client import search_queries
from .utils import cluster_videos
from itertools import count
from .schemas import ClusteredCandidate
import csv
import json
from .searchers.temporal import TemporalSearcher
from .searchers.ambiguous import AmbiguousSearcher
from json import JSONDecodeError

from .constants import MAP_KEYFRAMES_CSV_PATH, OBJECT_DETECTIONS_JSON_PATH

from adaptive_search.boundary_refinement import refine_event_boundary
from adaptive_search.client import normalize_video_id, parse_display_timestamp
from adaptive_search.dependencies import boundary_refinement_runtime
from adaptive_search.exceptions import UpstreamSearchError
from adaptive_search.providers import VideoAssetNotFoundError


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
    required_object_names = set(object_name_list or [])
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

            detection_class_names = object_data.get("detection_class_names", [])
            detection_scores = object_data.get("detection_scores", [])
            detected_object_names = {
                detection_class_name
                for detection_class_name, detection_score in zip(detection_class_names, detection_scores)
                if float(detection_score) >= objectThreshold
            }

            result.satisfiedObjects = required_object_names.issubset(detected_object_names)


def _refine_tuple_candidates(
    tuple_: list[ClusteredCandidate],
    *,
    video_name: str,
    query: list[str],
    capability_available: bool,
) -> list[dict]:
    """Boundary-refine every candidate in one result tuple, or mark why not.

    Legacy queries carry no pre/post-state text, so every call below passes
    boundary_type="unknown", pre_state=None, post_state=None - this dispatches
    refine_event_boundary() to the "state" (anchor-persistence) profile, never
    the degenerate transition path (see boundary_refinement.py's docstring).
    Each ClusteredCandidate's own `timestamp` (upstream-formatted "MM:SS", not
    `frame_index`, which is a keyframe index unrelated to real fps) is the
    anchor to refine from - legacy has no region/candidate-pool structure to
    pick a seed out of, so boundary_seeds.select_event_seeds() does not apply.
    """

    if not capability_available:
        return [
            {**candidate.model_dump(), "boundary_refinement_status": "skipped_runtime_unavailable"}
            for candidate in tuple_
        ]
    try:
        video_id = normalize_video_id(video_name)
    except UpstreamSearchError:
        return [
            {**candidate.model_dump(), "boundary_refinement_status": "skipped_video_not_in_catalog"}
            for candidate in tuple_
        ]
    metadata = boundary_refinement_runtime.frame_provider.catalog.metadata(video_id)
    if metadata is None or metadata.fps is None:
        return [
            {**candidate.model_dump(), "boundary_refinement_status": "skipped_no_metadata"}
            for candidate in tuple_
        ]

    refined = []
    for candidate in tuple_:
        dumped = candidate.model_dump()
        if candidate.query_id is None or not (0 <= candidate.query_id < len(query)):
            # Both searchers always stamp a valid query_id via cluster_videos
            # before a candidate can reach a result tuple; this is a defensive
            # fallback for the Optional[int] the schema still allows.
            dumped["boundary_refinement_status"] = "skipped_no_metadata"
            refined.append(dumped)
            continue
        try:
            outcome = refine_event_boundary(
                provider=boundary_refinement_runtime.frame_provider,
                embedder=boundary_refinement_runtime.embedder,
                video_id=video_id,
                event_id=f"legacy-query-{candidate.query_id}",
                anchor_query=query[candidate.query_id],
                pre_state=None,
                post_state=None,
                boundary_type="unknown",
                anchor_seconds=parse_display_timestamp(candidate.timestamp),
                fps=metadata.fps,
                duration_ms=metadata.duration_ms,
            )
        except VideoAssetNotFoundError:
            dumped["boundary_refinement_status"] = "skipped_video_not_in_catalog"
            refined.append(dumped)
            continue
        dumped["refined_timestamp_seconds"] = outcome.refined_seconds
        dumped["boundary_refinement_status"] = "applied"
        refined.append(dumped)
    return refined


def temporal_search(query: list[str], top_k_tuple: int, top_k_each_query: int, gamma: float,
                    searcher_type: str = "TemporalSearcher", objectFilterMode: bool = False, object_name_list: list[str] = None,
                    objectThreshold: float = 0.5, apply_boundary_refinement: bool = False
                    ) -> tuple[list[dict], dict]:
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

    capability_available = False
    capability_reason = None
    if apply_boundary_refinement:
        capability = boundary_refinement_runtime.capabilities()
        capability_available = capability.available
        capability_reason = capability.reason

    results = [
        {
            "score": score,
            "video_name": video_name,
            "tuple": (
                _refine_tuple_candidates(
                    tuple_,
                    video_name=video_name,
                    query=query,
                    capability_available=capability_available,
                )
                if apply_boundary_refinement
                else [
                    {**candidate.model_dump(), "boundary_refinement_status": "not_requested"}
                    for candidate in tuple_
                ]
            ),
        }
        for score, _, video_name, tuple_ in sorted(query_results, key=lambda x: x[0], reverse=True)
    ]
    boundary_refinement_capability = {
        "requested": apply_boundary_refinement,
        "available": capability_available,
        "reason": capability_reason,
    }
    return results, boundary_refinement_capability

if __name__ == "__main__":
    query_results, _capability = temporal_search(["cat", "dog", "bird", "fish"], 100, 100, 0.05, "TemporalSearcher"
                                    , objectFilterMode = True, object_name_list = ["cat", "dog"], objectThreshold = 0.5)
    ### export the result to a json file
    with open("data/temporal_search_results_temporal.json", "w", encoding="utf-8") as f:
        json.dump(query_results, f, ensure_ascii=False, indent=4)