
from typing import Dict 
import json
from video_clustering_schema import ClusteredCandidate, Videos


def cluster_videos(candidate_frames: list) -> list:
    videos: Dict[str, list] = {}
    for id, query in enumerate(candidate_frames):
        for frame in query.results:
            if frame.video_name not in videos:
                videos[frame.video_name] = []
            modified_frame = ClusteredCandidate(**frame.model_dump())
            modified_frame.query_id = id
            videos[frame.video_name].append(modified_frame)
    for video_name, frames in videos.items():
        frames.sort(key=lambda x: (x.frame_index, x.query_id))
    return [Videos(video_name=video_name, results=frames) for video_name, frames in videos.items()]

if __name__ == "__main__":
    from sendRequests import search_queries
    query0 = "cat"
    query1 = "dog"
    top_k = 5
    result = search_queries([query0, query1], top_k)
    clustered_videos = cluster_videos(result)
    print(clustered_videos)
    ## dump to json file
    with open("data/clustered_videos.json", "w", encoding="utf-8") as f:
        json.dump(
            [video.model_dump() for video in clustered_videos], 
            f, 
            indent=4,             
            ensure_ascii=False    
        )