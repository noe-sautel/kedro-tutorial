"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro_tutorial.pipelines.hello_modular import pipeline as hello_modular
from kedro_tutorial.pipelines.data_science import pipeline as data_science
from kedro_tutorial.pipelines.data_processing import pipeline as data_processing
from kedro.pipeline import Pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # pipelines = find_pipelines()
    # pipelines["__default__"] = sum(pipelines.values())
    
    hello_modular_pipeline = hello_modular.create_pipeline()
    data_science_pipeline = data_science.create_pipeline()
    data_processing_pipeline = data_processing.create_pipeline()

    return {
    "__default__": data_science_pipeline + data_processing_pipeline + hello_modular_pipeline,
    "hello_modular_pipeline": hello_modular_pipeline,
    }
