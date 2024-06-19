from typing import Union
from dataclasses import dataclass,asdict,field

@dataclass
class BasicMetricConfig:
    name:str = "basic_metrics_config"

    requieres_paths:bool = False
    requieres_origin:bool = False
    requieres_test_loop:bool = False

    number_of_samples_to_gather:int|str = 0
    compute_in_gpu:bool = False

#================================================
# MUSIC METRICS
#================================================

@dataclass
class HellingerMetricConfig(BasicMetricConfig):
    name:str = "histogram_hellinger"

    binary:bool = False
    plot_histogram:bool = False
    plot_binary_histogram:bool = False

@dataclass
class OutlierMetricConfig(BasicMetricConfig):
    name:str = "outlier"

@dataclass
class MusicPlotConfig(BasicMetricConfig):
    name:str = "music_plot"
    number_of_samples_to_gather:int = 10
    requieres_origin:bool = True

@dataclass
class GraphMetricsConfig(BasicMetricConfig):
    name: str = "graphs_metrics"
    number_of_samples_to_gather: str = "all"
    requieres_origin:bool = True
    plot_graphs: bool = False
    windows:bool = False
    methods: list[str] = field(default_factory=lambda: ["degree", "cluster", "orbit"])

@dataclass
class MetricsAvaliable:
    histogram_hellinger:str = "histogram_hellinger"
    music_plot:str = "music_plot"

metrics_config = {
    "histogram_hellinger":HellingerMetricConfig,
    "music_plot":MusicPlotConfig,
    "graphs_metrics":GraphMetricsConfig,
}