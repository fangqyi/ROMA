REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .sc_episode_runner import SCEpisodeRunner
REGISTRY["sc_episode"] = SCEpisodeRunner
