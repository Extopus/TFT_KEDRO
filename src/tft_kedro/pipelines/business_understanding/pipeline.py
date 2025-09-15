from kedro.pipeline import Pipeline, node
from .nodes import describe_dataset


def create_pipeline(**kwargs) -> Pipeline:
    """Create the business understanding pipeline."""
    return Pipeline([
        node(
            func=describe_dataset,
            inputs="challenger_raw",
            outputs="challenger_stats",
            name="describe_challenger"
        ),
        node(
            func=describe_dataset,
            inputs="platinum_raw",
            outputs="platinum_stats",
            name="describe_platinum"
        ),
        node(
            func=describe_dataset,
            inputs="grandmaster_raw",
            outputs="grandmaster_stats",
            name="describe_grandmaster"
        )
    ])
