"""athanor.hypotheses — hypothesis generation and experiment design (Stage 3)."""
from athanor.hypotheses.models import Hypothesis, ExperimentDesign, HypothesisReport
from athanor.hypotheses.generator import HypothesisGenerator
from athanor.hypotheses.critic import HypothesisCritic

__all__ = ["Hypothesis", "ExperimentDesign", "HypothesisReport", "HypothesisGenerator", "HypothesisCritic"]
