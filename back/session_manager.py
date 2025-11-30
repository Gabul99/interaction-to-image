"""
Session Management Module
"""
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import uuid
import sys
import os
from fastapi import WebSocket

try:
    from .models import ObjectChip, BoundingBox, FeedbackRecord
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from models import ObjectChip, BoundingBox, FeedbackRecord


class SessionStatus(str, Enum):
    """Session status"""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Session:
    """Session data"""
    id: str
    prompt: str
    objects: List[ObjectChip]
    bboxes: List[BoundingBox]
    width: int
    height: int
    num_inference_steps: int
    status: SessionStatus = SessionStatus.PENDING
    root_node_id: Optional[str] = None
    websocket: Optional[WebSocket] = None
    current_step: int = 0
    feedbacks: List[FeedbackRecord] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: __import__('time').time())


@dataclass
class Branch:
    """Branch data"""
    id: str
    session_id: str
    source_node_id: str
    feedback: List[FeedbackRecord]
    websocket: Optional[WebSocket] = None
    current_step: int = 0
    created_at: float = field(default_factory=lambda: __import__('time').time())


class SessionManager:
    """Session manager"""
    
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._branches: Dict[str, Branch] = {}
    
    def create_session(
        self,
        prompt: str,
        objects: List[ObjectChip],
        bboxes: List[BoundingBox],
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 50
    ) -> Session:
        """Create a new session"""
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        root_node_id = f"node_prompt_{uuid.uuid4().hex[:12]}"
        
        session = Session(
            id=session_id,
            prompt=prompt,
            objects=objects or [],
            bboxes=bboxes or [],
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            root_node_id=root_node_id,
        )
        
        self._sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        return self._sessions.get(session_id)
    
    def set_websocket(self, session_id: str, websocket: WebSocket):
        """Set WebSocket for session"""
        session = self._sessions.get(session_id)
        if session:
            session.websocket = websocket
    
    def update_session_status(self, session_id: str, status: SessionStatus):
        """Update session status"""
        session = self._sessions.get(session_id)
        if session:
            session.status = status
    
    def update_session_step(self, session_id: str, step: int):
        """Update session step"""
        session = self._sessions.get(session_id)
        if session:
            session.current_step = step
    
    def add_feedback(self, session_id: str, feedback: FeedbackRecord):
        """Add feedback to session"""
        session = self._sessions.get(session_id)
        if session:
            session.feedbacks.append(feedback)
    
    def get_feedbacks(self, session_id: str) -> List[FeedbackRecord]:
        """Get feedback list for session"""
        session = self._sessions.get(session_id)
        if session:
            return session.feedbacks.copy()
        return []
    
    def clear_feedbacks(self, session_id: str):
        """Clear feedback list for session"""
        session = self._sessions.get(session_id)
        if session:
            session.feedbacks.clear()
    
    def create_branch(
        self,
        session_id: str,
        source_node_id: str,
        feedback: List[FeedbackRecord]
    ) -> Branch:
        """Create a new branch"""
        branch_id = f"branch_{uuid.uuid4().hex[:12]}"
        
        branch = Branch(
            id=branch_id,
            session_id=session_id,
            source_node_id=source_node_id,
            feedback=feedback,
        )
        
        self._branches[branch_id] = branch
        return branch
    
    def get_branch(self, branch_id: str) -> Optional[Branch]:
        """Get branch by ID"""
        return self._branches.get(branch_id)
    
    def set_branch_websocket(self, branch_id: str, websocket: WebSocket):
        """Set WebSocket for branch"""
        branch = self._branches.get(branch_id)
        if branch:
            branch.websocket = websocket
    
    def update_branch_step(self, branch_id: str, step: int):
        """Update branch step"""
        branch = self._branches.get(branch_id)
        if branch:
            branch.current_step = step
