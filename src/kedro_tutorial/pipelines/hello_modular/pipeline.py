"""
This is a boilerplate pipeline 'hello_modular'
generated using Kedro 0.18.3
"""

from kedro.pipeline import pipeline, node

from kedro_tutorial.pipelines.hello_modular.nodes import get_mean


def create_pipeline(**kwargs):
    return pipeline(
        [
        node(
            func=get_mean,
            inputs= ["companies", "params:selected_columns.companies",],
            outputs=None,
            name="get_mean",
        ),
        ]
                    )