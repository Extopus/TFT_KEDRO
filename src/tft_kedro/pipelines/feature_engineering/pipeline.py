from kedro.pipeline import Pipeline, node
from .nodes import add_features, combine_features


def create_pipeline(**kwargs) -> Pipeline:
    """Create the feature engineering pipeline."""
    return Pipeline([
        node(
        func=add_features,
        inputs="challenger_intermediate",
        outputs="challenger_features",
        name="features_challenger"
        ),
        node(
        func=add_features,
        inputs="platinum_intermediate",
        outputs="platinum_features",
        name="features_platinum"
        ),
        node(
        func=add_features,
        inputs="grandmaster_intermediate",
        outputs="grandmaster_features",
        name="features_grandmaster"
        ),
        node(
        func=combine_features,
        inputs=["challenger_features", "platinum_features", "grandmaster_features"],
        outputs="tft_combined_features",
        name="combine_all_features"
        )
    ])
