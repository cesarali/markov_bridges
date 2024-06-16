
from dataclasses import dataclass,asdict,field

@dataclass
class BasicMetricConfig:
    name:str = "histogram_hellinger"

    requieres_paths:bool = False
    requieres_origin:bool = False
    requieres_test_loop:bool = False

#================================================
# MUSIC METRICS
#================================================

@dataclass
class HellingerMetricConfig(BasicMetricConfig):
    name:str = "histogram_hellinger"

    requieres_paths:bool = False
    requieres_origin:bool = False
    requieres_test_loop:bool = False

    plot_binary_histogram:bool = False

@dataclass
class OutlierMetricConfig(BasicMetricConfig):
    name:str = "outlier"

    requieres_paths:bool = False
    requieres_origin:bool = False
    requieres_test_loop:bool = False

    plot_binary_histogram:bool = False

@dataclass
class MusicPlotConfig(BasicMetricConfig):
    name:str = "music_plot"

    requieres_paths:bool = False
    requieres_origin:bool = False
    requieres_test_loop:bool = False

    plot_binary_histogram:bool = False

@dataclass
class MetricsAvaliable:
    histogram_hellinger:str = "histogram_hellinger"
    outlier:str = "outlier"
    music_plot:str = "music_plot"

metrics_config = {
    "histogram_hellinger":HellingerMetricConfig,
    "outlier":OutlierMetricConfig,
    "music_plot":MusicPlotConfig,
}