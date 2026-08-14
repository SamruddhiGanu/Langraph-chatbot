# graph/blog/schemas.py — Provider-agnostic Pydantic schemas for the blog pipeline

from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Router decision
# ---------------------------------------------------------------------------
class RouterDecision(BaseModel):
    """Routing decision: does this topic need web research before planning?"""
    mode: Literal["closed_book", "hybrid", "open_book"] = Field(
        description=(
            "closed_book = evergreen concept, no web research needed; "
            "hybrid = mostly evergreen but needs some current examples/tools; "
            "open_book = highly time-sensitive, needs latest web data"
        )
    )
    needs_research: bool = Field(
        description="True if web/PDF research should be run before planning."
    )
    queries: List[str] = Field(
        default_factory=list,
        description="3-10 high-signal search queries to run if needs_research=True."
    )


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------
class EvidenceItem(BaseModel):
    """A single piece of research evidence (web or PDF-sourced)."""
    title: str = Field(default="", description="Title of the source.")
    url: str = Field(description="URL of the web source, or 'uploaded_pdf' for PDF evidence.")
    snippet: str = Field(description="Relevant excerpt (≤600 chars).")
    published_at: Optional[str] = Field(
        default=None, description="ISO date string YYYY-MM-DD, or null if unknown."
    )
    source: Literal["web", "uploaded_pdf"] = Field(
        default="web",
        description="'web' for Tavily results, 'uploaded_pdf' for PDF-extracted content."
    )


class EvidencePack(BaseModel):
    """Container returned by the evidence-extraction LLM call."""
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Blog plan
# ---------------------------------------------------------------------------
class Task(BaseModel):
    """A single section task for the blog worker."""
    section_title: str = Field(description="H2 heading for this section.")
    section_index: int = Field(description="Order of this section (0-indexed).")
    guidelines: str = Field(
        description="Specific writing instructions for this section (1-3 sentences)."
    )
    relevant_evidence_urls: List[str] = Field(
        default_factory=list,
        description="URLs (or 'uploaded_pdf') of evidence items relevant to this section."
    )


class Plan(BaseModel):
    """The orchestrator's blog plan."""
    title: str = Field(description="Full blog post title.")
    intro_guidelines: str = Field(description="Instructions for the introduction section.")
    tasks: List[Task] = Field(description="List of body section tasks (excluding intro/conclusion).")
    conclusion_guidelines: str = Field(description="Instructions for the conclusion section.")
    target_word_count: int = Field(default=1200, description="Approximate total word count target.")


# ---------------------------------------------------------------------------
# Image planning
# ---------------------------------------------------------------------------
class ImageSpec(BaseModel):
    """Specification for one image to generate and place in the blog."""
    placeholder: str = Field(
        description="Exact placeholder string in the markdown, e.g. '{{IMAGE_1}}'."
    )
    prompt: str = Field(description="Detailed image generation prompt (≤300 chars).")
    alt_text: str = Field(description="Alt text for the image.")
    placement_hint: str = Field(
        description="Which section or heading the image should appear near."
    )


class GlobalImagePlan(BaseModel):
    """Image plan returned by the decide_images node."""
    images: List[ImageSpec] = Field(
        default_factory=list,
        description="1-3 images to generate. Keep it ≤3 to control cost."
    )
