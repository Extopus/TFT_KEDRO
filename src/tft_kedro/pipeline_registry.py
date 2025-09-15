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
    from tft_kedro.pipelines import business_understanding
    from tft_kedro.pipelines import feature_engineering
    from tft_kedro.pipelines import data_science
    from tft_kedro.pipelines import reporting
    pipelines = find_pipelines()
    pipelines["data_cleaning"] = data_cleaning.create_pipeline()
    pipelines["business_understanding"] = business_understanding.create_pipeline()
    pipelines["feature_engineering"] = feature_engineering.create_pipeline()
    pipelines["data_science"] = data_science.create_pipeline()
    pipelines["reporting"] = reporting.create_pipeline()
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
