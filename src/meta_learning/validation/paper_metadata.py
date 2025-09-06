"""
Paper Metadata Management for Research Validation System.

This module provides comprehensive metadata storage and management
for research papers used in validation processes.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PaperMetadata:
    """Structured metadata for a research paper."""
    
    paper_title: str
    authors: str  
    year: int
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    key_contributions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Generate citation string after initialization."""
        self.citation = f"{self.authors} ({self.year}). {self.paper_title}"
        if self.venue:
            self.citation += f" In {self.venue}."


@dataclass  
class MathematicalFormulation:
    """Mathematical equation or formulation from paper."""
    
    name: str
    equation: str
    description: str
    section: Optional[str] = None
    equation_number: Optional[str] = None
    variables: Dict[str, str] = field(default_factory=dict)
    
    def __str__(self):
        return f"{self.name}: {self.equation}"


@dataclass
class BenchmarkResult:
    """Benchmark result reported in paper."""
    
    dataset: str
    metric: str
    value: float
    setting: str  # e.g., "5-way 1-shot", "10-way 5-shot"
    confidence_interval: Optional[Tuple[float, float]] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        ci_str = ""
        if self.confidence_interval:
            ci_str = f" ± {(self.confidence_interval[1] - self.confidence_interval[0]) / 2:.3f}"
        return f"{self.dataset} ({self.setting}): {self.value:.3f}{ci_str} {self.metric}"


class PaperDatabase:
    """Database of research papers for validation."""
    
    def __init__(self):
        """Initialize paper database."""
        self.papers = {}
        self.logger = logging.getLogger("research_validation.paper_db")
        
    def add_paper(self, paper_id: str, metadata: PaperMetadata, 
                  equations: List[MathematicalFormulation],
                  benchmarks: List[BenchmarkResult]) -> None:
        """Add paper to database."""
        self.papers[paper_id] = {
            'metadata': metadata,
            'equations': {eq.name: eq for eq in equations},
            'benchmarks': {f"{b.dataset}_{b.setting}_{b.metric}": b for b in benchmarks}
        }
        self.logger.info(f"Added paper: {metadata.citation}")
        
    def get_paper(self, paper_id: str) -> Optional[Dict]:
        """Retrieve paper by ID."""
        return self.papers.get(paper_id)
        
    def list_papers(self) -> List[str]:
        """List all paper IDs."""
        return list(self.papers.keys())
        
    def find_papers_by_author(self, author: str) -> List[str]:
        """Find papers by author name."""
        results = []
        for paper_id, paper_data in self.papers.items():
            if author.lower() in paper_data['metadata'].authors.lower():
                results.append(paper_id)
        return results
        
    def find_papers_by_year(self, year: int) -> List[str]:
        """Find papers by publication year."""
        results = []
        for paper_id, paper_data in self.papers.items():
            if paper_data['metadata'].year == year:
                results.append(paper_id)
        return results


# Pre-defined research papers for meta-learning validation
def create_maml_paper() -> Tuple[str, PaperMetadata, List[MathematicalFormulation], List[BenchmarkResult]]:
    """Create MAML paper entry."""
    
    metadata = PaperMetadata(
        paper_title="Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks",
        authors="Finn et al.",
        year=2017,
        venue="ICML",
        key_contributions=[
            "Model-agnostic meta-learning algorithm",
            "First-order approximation (FOMAML)",
            "Gradient-based meta-learning framework"
        ]
    )
    
    equations = [
        MathematicalFormulation(
            name="inner_update",
            equation="θ'_i = θ - α∇_θ L_{T_i}(f_θ)",
            description="Inner loop parameter update for task T_i",
            section="3.1",
            equation_number="(1)",
            variables={
                "θ": "model parameters",
                "α": "inner learning rate", 
                "L_{T_i}": "task-specific loss function",
                "f_θ": "model with parameters θ"
            }
        ),
        MathematicalFormulation(
            name="meta_gradient",
            equation="θ ← θ - β∇_θ Σ_i L_{T_i}(f_{θ'_i})",
            description="Meta-update using adapted parameters",
            section="3.1", 
            equation_number="(2)",
            variables={
                "β": "meta learning rate",
                "θ'_i": "adapted parameters for task i"
            }
        )
    ]
    
    benchmarks = [
        BenchmarkResult(
            dataset="Omniglot",
            metric="accuracy",
            value=0.989,
            setting="5-way 1-shot",
            confidence_interval=(0.985, 0.993)
        ),
        BenchmarkResult(
            dataset="Omniglot", 
            metric="accuracy",
            value=0.997,
            setting="5-way 5-shot",
            confidence_interval=(0.995, 0.999)
        ),
        BenchmarkResult(
            dataset="miniImageNet",
            metric="accuracy", 
            value=0.487,
            setting="5-way 1-shot",
            confidence_interval=(0.463, 0.511)
        ),
        BenchmarkResult(
            dataset="miniImageNet",
            metric="accuracy",
            value=0.639,
            setting="5-way 5-shot", 
            confidence_interval=(0.611, 0.667)
        )
    ]
    
    return "maml_finn2017", metadata, equations, benchmarks


def create_protonet_paper() -> Tuple[str, PaperMetadata, List[MathematicalFormulation], List[BenchmarkResult]]:
    """Create Prototypical Networks paper entry."""
    
    metadata = PaperMetadata(
        paper_title="Prototypical Networks for Few-shot Learning", 
        authors="Snell et al.",
        year=2017,
        venue="NeurIPS",
        key_contributions=[
            "Prototypical networks for few-shot learning",
            "Non-parametric distance-based classification",
            "Simple yet effective approach to few-shot learning"
        ]
    )
    
    equations = [
        MathematicalFormulation(
            name="prototype_computation",
            equation="c_k = (1/|S_k|) Σ_{(x_i,y_i)∈S_k} f_φ(x_i)",
            description="Prototype computation as mean of support set embeddings",
            section="3.1",
            equation_number="(1)",
            variables={
                "c_k": "prototype for class k",
                "S_k": "support set for class k", 
                "f_φ": "embedding function",
                "φ": "embedding parameters"
            }
        ),
        MathematicalFormulation(
            name="distance_function",
            equation="d(f_φ(x), c_k) = ||f_φ(x) - c_k||_2^2",
            description="Squared Euclidean distance to prototype",
            section="3.1",
            equation_number="(2)",
            variables={
                "d": "distance function",
                "||·||_2": "L2 norm"
            }
        ),
        MathematicalFormulation(
            name="softmax_probability", 
            equation="p_φ(y=k|x) = exp(-d(f_φ(x), c_k)) / Σ_{k'} exp(-d(f_φ(x), c_{k'}))",
            description="Softmax probability over prototype distances",
            section="3.1",
            equation_number="(3)"
        )
    ]
    
    benchmarks = [
        BenchmarkResult(
            dataset="Omniglot",
            metric="accuracy",
            value=0.985,
            setting="5-way 1-shot"
        ),
        BenchmarkResult(
            dataset="Omniglot",
            metric="accuracy", 
            value=0.988,
            setting="5-way 5-shot"
        ),
        BenchmarkResult(
            dataset="miniImageNet",
            metric="accuracy",
            value=0.494,
            setting="5-way 1-shot"
        ),
        BenchmarkResult(
            dataset="miniImageNet",
            metric="accuracy",
            value=0.678,
            setting="5-way 5-shot"
        )
    ]
    
    return "protonet_snell2017", metadata, equations, benchmarks


def create_metasgd_paper() -> Tuple[str, PaperMetadata, List[MathematicalFormulation], List[BenchmarkResult]]:
    """Create Meta-SGD paper entry."""
    
    metadata = PaperMetadata(
        paper_title="Meta-SGD: Learning to Learn Quickly for Few Shot Learning",
        authors="Li et al.", 
        year=2017,
        venue="ArXiv",
        key_contributions=[
            "Meta-SGD with learnable learning rates",
            "Per-parameter learning rate adaptation", 
            "Improved gradient-based meta-learning"
        ]
    )
    
    equations = [
        MathematicalFormulation(
            name="meta_sgd_update",
            equation="θ'_i = θ - α ⊙ ∇_θ L_{T_i}(f_θ)",
            description="Meta-SGD inner update with element-wise learning rates",
            section="3.2",
            equation_number="(4)",
            variables={
                "α": "learnable per-parameter learning rates",
                "⊙": "element-wise multiplication"
            }
        ),
        MathematicalFormulation(
            name="learning_rate_meta_update",
            equation="α ← α - β_α∇_α Σ_i L_{T_i}(f_{θ'_i})",
            description="Meta-update for learning rates",
            section="3.2",
            equation_number="(5)",
            variables={
                "β_α": "meta learning rate for α parameters"
            }
        )
    ]
    
    benchmarks = [
        BenchmarkResult(
            dataset="Omniglot",
            metric="accuracy",
            value=0.993,
            setting="5-way 1-shot"
        ),
        BenchmarkResult(
            dataset="miniImageNet", 
            metric="accuracy",
            value=0.508,
            setting="5-way 1-shot"
        )
    ]
    
    return "metasgd_li2017", metadata, equations, benchmarks


def initialize_paper_database() -> PaperDatabase:
    """Initialize database with standard meta-learning papers."""
    db = PaperDatabase()
    
    # Add MAML paper
    paper_id, metadata, equations, benchmarks = create_maml_paper()
    db.add_paper(paper_id, metadata, equations, benchmarks)
    
    # Add ProtoNet paper
    paper_id, metadata, equations, benchmarks = create_protonet_paper()
    db.add_paper(paper_id, metadata, equations, benchmarks)
    
    # Add Meta-SGD paper
    paper_id, metadata, equations, benchmarks = create_metasgd_paper()
    db.add_paper(paper_id, metadata, equations, benchmarks)
    
    return db