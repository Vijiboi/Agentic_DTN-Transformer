from __future__ import annotations

from math import log10, log2, sqrt

from .config import SimulationConfig
from .models import APNode, Task


def distance_km(task: Task, ap: APNode) -> float:
    distance_m = sqrt((task.x - ap.x) ** 2 + (task.y - ap.y) ** 2 + (task.z - ap.z) ** 2)
    return max(distance_m / 1000.0, 1e-6)


def free_space_path_loss_db(task: Task, ap: APNode, config: SimulationConfig) -> float:
    frequency_ghz = config.carrier_frequency_ghz_by_tier[ap.tier]
    return 32.45 + 20.0 * log10(frequency_ghz) + 20.0 * log10(distance_km(task, ap))


def large_scale_gain_from_path_loss(path_loss_db: float) -> float:
    return 10.0 ** (-path_loss_db / 10.0)


def predicted_uplink_rate(task: Task, ap: APNode, config: SimulationConfig, load_reference: float) -> float:
    """Implements the paper rate model where enough paper parameters are available.

    The paper gives the free-space path-loss instantiation for HAP/LEO links.
    For BS links it only states a terrestrial path-loss model, so the current
    bandwidth-normalized fallback is kept instead of inventing a macro-cell law.
    """
    task_count = max(load_reference, 1.0)
    allocated_bandwidth = max(ap.bandwidth / task_count, 1e-9)

    if ap.tier == "BS":
        return allocated_bandwidth

    path_loss = free_space_path_loss_db(task, ap, config)
    gain = large_scale_gain_from_path_loss(path_loss)
    snr = config.uav_transmit_power * gain / max(config.noise_spectral_density * allocated_bandwidth, 1e-18)
    return max(allocated_bandwidth * log2(1.0 + snr), 1e-9)
