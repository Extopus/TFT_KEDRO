from kedro.pipeline import Pipeline, node
from .nodes import clean_csv_data

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=clean_csv_data,
            inputs="raw_csv",
            outputs="cleaned_csv",
            name="clean_csv_data_node"
        )
    ])
