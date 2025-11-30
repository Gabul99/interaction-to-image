"""
Backend API Models
"""
from pydantic import BaseModel
from typing import List, Optional, Literal
from enum import Enum


class ObjectChip(BaseModel):
    """Object chip model"""
    id: str
    label: str
    color: str


class BoundingBox(BaseModel):
    """Bounding box model (frontend format: relative coordinates 0~1)"""
    objectId: str
    x: float
    y: float
    width: float
    height: float


class ImageGenerationRequest(BaseModel):
    """Image generation request"""
    prompt: str
    objects: Optional[List[ObjectChip]] = []
    bboxes: Optional[List[BoundingBox]] = []
    width: int = 512
    height: int = 512
    num_inference_steps: int = 50


class FeedbackArea(str, Enum):
    """Feedback area type"""
    full = "full"
    bbox = "bbox"
    point = "point"


class FeedbackType(str, Enum):
    """Feedback type"""
    text = "text"
    image = "image"


class FeedbackRecord(BaseModel):
    """Feedback record"""
    id: str
    area: FeedbackArea
    type: FeedbackType
    text: Optional[str] = None
    imageUrl: Optional[str] = None
    point: Optional[dict] = None
    bbox: Optional[dict] = None
    bboxId: Optional[str] = None
    timestamp: int


class BranchCreateRequest(BaseModel):
    """Branch creation request"""
    sessionId: str
    sourceNodeId: str
    feedback: List[FeedbackRecord]


class WebSocketMessageType(str, Enum):
    """WebSocket message type"""
    image_step = "image_step"
    generation_complete = "generation_complete"
    error = "error"


class WebSocketMessage(BaseModel):
    """WebSocket message"""
    type: WebSocketMessageType
    sessionId: str
    nodeId: Optional[str] = None
    branchId: Optional[str] = None
    parentNodeId: Optional[str] = None
    step: Optional[int] = None
    totalSteps: Optional[int] = None
    imageUrl: Optional[str] = None
    imageData: Optional[str] = None
    timestamp: Optional[int] = None
    message: Optional[str] = None
