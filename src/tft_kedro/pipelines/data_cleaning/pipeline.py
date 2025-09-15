from kedro.pipeline import Pipeline, node
from .nodes import clean_csv_data

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=clean_csv_data,
            inputs="challenger_raw",
            outputs="challenger_intermediate",
            name="clean_challenger"
        ),
        node(
            func=clean_csv_data,
            inputs="platinum_raw",
            outputs="platinum_intermediate",
            name="clean_platinum"
        ),
        node(
            func=clean_csv_data,
            inputs="grandmaster_raw",
            outputs="grandmaster_intermediate",
            name="clean_grandmaster"
        )
    ])
