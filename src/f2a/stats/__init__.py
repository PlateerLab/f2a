"""Stats module — statistical analysis engine."""

from f2a.stats.categorical import CategoricalStats
from f2a.stats.correlation import CorrelationStats
from f2a.stats.descriptive import DescriptiveStats
from f2a.stats.distribution import DistributionStats
from f2a.stats.duplicates import DuplicateStats
from f2a.stats.feature_importance import FeatureImportanceStats
from f2a.stats.missing import MissingStats
from f2a.stats.outlier import OutlierStats
from f2a.stats.pca_analysis import PCAStats
from f2a.stats.quality import QualityStats

__all__ = [
    "CategoricalStats",
    "CorrelationStats",
    "DescriptiveStats",
    "DistributionStats",
    "DuplicateStats",
    "FeatureImportanceStats",
    "MissingStats",
    "OutlierStats",
    "PCAStats",
    "QualityStats",
]
