from pydantic import BaseModel
from video_clustering_schema import ClusteredCandidate

class Candidate(ClusteredCandidate):
    video_name: str
    video_title: str
    author: str
    watch_url: str
class QueryResponse(BaseModel):
    query: str
    results: list[Candidate]  