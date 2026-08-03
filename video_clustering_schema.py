from pydantic import BaseModel
from typing import Optional

class ClusteredCandidate(BaseModel):
    frame_index: int
    timestamp: str
    score: float
    query_id: Optional[int] = None
    satisfiedObjects: Optional[bool] = None


class Videos(BaseModel):
    video_name: str
    results: list[ClusteredCandidate]
