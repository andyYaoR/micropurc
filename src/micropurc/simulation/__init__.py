"""Simulation.

A data-generating process (:class:`~micropurc.simulation.dgp.DGP`) that draws
synthetic trips from the microPURC model, and a flow-decomposition route
sampler (:class:`~micropurc.simulation.route_sampler.MarkovSampler`)
that turns link flows into sampled routes.
"""

from .dgp import DGP, DGPConfig
from .route_sampler import MarkovSampler, SampledRoute

__all__ = ["DGP", "DGPConfig", "MarkovSampler", "SampledRoute"]
