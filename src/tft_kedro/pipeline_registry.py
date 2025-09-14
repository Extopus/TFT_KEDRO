"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    from tft_kedro.pipelines import data_cleaning
    pipelines = find_pipelines()
    pipelines["data_cleaning"] = data_cleaning.create_pipeline()
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
