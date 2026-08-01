"""
FT-QuPAP Machine Learning Package.
"""

from .feature_schema import FEATURE_COLUMNS
from .gp_model_loader import (
    GPModelBundle,
    GPModelLoader,
    load_gp_model_bundle,
)
from .gp_predictor import (
    FTQuPAPGPPredictor,
    GPPrediction,
    predict_attack,
)
from .metrics import (
    GPMetricsReport,
    evaluate_gp_metrics,
)
from .model_evaluator import (
    FTQuPAPModelEvaluator,
    HeldOutEvaluation,
    IndependentEvaluation,
    ModelEvaluationResult,
)
from .model_trainer import (
    FTQuPAPModelTrainer,
    GPTrainingConfig,
    GPTrainingResult,
)
from .probability_calibrator import (
    ProbabilityCalibrator,
)
from .threshold_manager import (
    ThresholdManager,
    ThresholdPolicy,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "FEATURE_COLUMNS",
    "GPModelBundle",
    "GPModelLoader",
    "load_gp_model_bundle",
    "FTQuPAPGPPredictor",
    "GPPrediction",
    "predict_attack",
    "GPMetricsReport",
    "evaluate_gp_metrics",
    "FTQuPAPModelTrainer",
    "GPTrainingConfig",
    "GPTrainingResult",
    "FTQuPAPModelEvaluator",
    "HeldOutEvaluation",
    "IndependentEvaluation",
    "ModelEvaluationResult",
    "ProbabilityCalibrator",
    "ThresholdManager",
    "ThresholdPolicy",
]