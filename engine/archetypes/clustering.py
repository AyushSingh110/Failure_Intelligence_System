import math
import uuid
from dataclasses import dataclass, field
from typing import TypedDict

from app.schemas import FailureSignalVector
from engine.archetypes.similarity import compute_signal_similarity
from engine.archetypes.labeling import label_failure_archetype
from config import get_settings

settings = get_settings()

ARCHETYPE_NOVEL_ANOMALY = "NOVEL_ANOMALY"

_LOG_GROWTH_FACTOR = 0.03


def _adaptive_threshold(num_existing_clusters: int) -> float:
    """
    Logarithmic adaptive threshold.
    Grows quickly for the first ~10 clusters then plateaus.
    """
    base    = settings.cluster_base_similarity_threshold
    growth  = math.log(num_existing_clusters + 1) * _LOG_GROWTH_FACTOR
    return min(base + growth, settings.cluster_threshold_max)


class ClusterAssignment(TypedDict):
    cluster_id: str | None
    status: str        # "KNOWN_FAILURE" | "NOVEL_ANOMALY" | "AMBIGUOUS"
    similarity_score: float
    archetype: str


@dataclass
class FailureCluster:
    cluster_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    centroid:    FailureSignalVector = field(default=None)   # type: ignore[assignment]
    members:     list[FailureSignalVector] = field(default_factory=list)
    archetype:   str = ""
    is_novel:    bool = False
    created_at_size: int = 0   # cluster count at time of creation (diagnostic)

    def __post_init__(self) -> None:
        if self.centroid is not None and not self.members:
            self.members.append(self.centroid)
        if not self.archetype:
            self.archetype = (
                ARCHETYPE_NOVEL_ANOMALY if self.is_novel
                else label_failure_archetype(self.centroid)
            )

    def add_member(self, signal: FailureSignalVector) -> None:
        self.members.append(signal)
        # Promote a novel anomaly to a confirmed archetype once a second
        # signal joins — it is no longer unique.
        if self.is_novel and len(self.members) >= 2:
            self.is_novel = False
            self.archetype = label_failure_archetype(self.centroid)

    def size(self) -> int:
        return len(self.members)


class ArchetypeClusterRegistry:
    """
    Stateful registry of known failure clusters.
    Call .assign(signal) for each new signal in real time.
    """

    def __init__(self) -> None:
        self._clusters: list[FailureCluster] = []

    def cluster_count(self) -> int:
        return len(self._clusters)

    def assign(self, signal: FailureSignalVector) -> ClusterAssignment:
        """
        Assigns a signal to an existing cluster or creates a new one.
        Returns a ClusterAssignment describing the outcome.
        """
        if not self._clusters:
            cluster = FailureCluster(centroid=signal, is_novel=True, created_at_size=0)
            self._clusters.append(cluster)
            return ClusterAssignment(
                cluster_id=cluster.cluster_id,
                status="NOVEL_ANOMALY",
                similarity_score=1.0,
                archetype=cluster.archetype,
            )

        # Score against all existing centroids
        scored = [
            (cluster, compute_signal_similarity(signal, cluster.centroid))
            for cluster in self._clusters
        ]
        best_cluster, best_score = max(scored, key=lambda x: x[1])
        threshold = _adaptive_threshold(len(self._clusters))

        if best_score >= threshold:
            best_cluster.add_member(signal)
            return ClusterAssignment(
                cluster_id=best_cluster.cluster_id,
                status="KNOWN_FAILURE",
                similarity_score=best_score,
                archetype=best_cluster.archetype,
            )

        if best_score < settings.cluster_novel_anomaly_ceiling:
            new_cluster = FailureCluster(
                centroid=signal,
                is_novel=True,
                created_at_size=len(self._clusters),
            )
            self._clusters.append(new_cluster)
            return ClusterAssignment(
                cluster_id=new_cluster.cluster_id,
                status="NOVEL_ANOMALY",
                similarity_score=best_score,
                archetype=new_cluster.archetype,
            )

        # Ambiguous zone — distinct but not alien
        new_cluster = FailureCluster(
            centroid=signal,
            is_novel=False,
            created_at_size=len(self._clusters),
        )
        self._clusters.append(new_cluster)
        return ClusterAssignment(
            cluster_id=new_cluster.cluster_id,
            status="AMBIGUOUS",
            similarity_score=best_score,
            archetype=new_cluster.archetype,
        )

    def get_all_clusters(self) -> list[FailureCluster]:
        return sorted(self._clusters, key=lambda c: c.size(), reverse=True)

    def summarize(self) -> list[dict]:
        return [
            {
                "cluster_id":      c.cluster_id,
                "archetype":       c.archetype,
                "cluster_size":    c.size(),
                "is_novel_anomaly": c.is_novel,
                "centroid":        c.centroid.model_dump(),
            }
            for c in self.get_all_clusters()
        ]


# ── Stateless batch API (for one-shot analysis without a registry) ────────

def cluster_signals(
    signals: list[FailureSignalVector],
) -> list[FailureCluster]:
    """
    Processes a batch of signals through a fresh registry.
    Returns all resulting clusters sorted by size descending.
    """
    registry = ArchetypeClusterRegistry()
    for signal in signals:
        registry.assign(signal)
    return registry.get_all_clusters()


def summarize_clusters(clusters: list[FailureCluster]) -> list[dict]:
    return [
        {
            "cluster_id":       c.cluster_id,
            "archetype":        c.archetype,
            "cluster_size":     c.size(),
            "is_novel_anomaly": c.is_novel,
            "centroid":         c.centroid.model_dump(),
        }
        for c in clusters
    ]


# Singleton registry for real-time use in the FastAPI app
archetype_registry = ArchetypeClusterRegistry()
