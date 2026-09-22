import sys
from pathlib import Path

# Make the repo root (parent of this tests/ folder) importable so
# `from components.ingest import ingest_data` etc. works when running
# `pytest` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeOutput:
    """Stand-in for a KFP Output[Dataset]/Output[Model] artifact.

    The real KFP objects carry a lot more (URIs, metadata store links), but
    every component under test only ever touches `.path`, so that's all we
    need here.
    """

    def __init__(self, path):
        self.path = str(path)


class FakeMetrics:
    """Stand-in for a KFP Output[Metrics] artifact.

    Deliberately NOT using kfp.dsl.Metrics() here: its constructor signature
    has varied across kfp versions, which is exactly the kind of environment
    fragility a unit test should avoid depending on. This just mimics the
    two things components/train.py actually calls: .log_metric() and
    .metadata.
    """

    def __init__(self):
        self.metadata = {}

    def log_metric(self, key, value):
        self.metadata[key] = value