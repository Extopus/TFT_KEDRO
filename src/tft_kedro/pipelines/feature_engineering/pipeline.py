from kedro.pipeline import Pipeline, node
from .nodes import add_features

pipeline = Pipeline([
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
    )
])
