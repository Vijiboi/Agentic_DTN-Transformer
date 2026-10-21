from __future__ import annotations

from math import sqrt

from .models import APNode, UAVAgent


def euclidean_distance_3d(
    x1: float,
    y1: float,
    z1: float,
    x2: float,
    y2: float,
    z2: float,
) -> float:
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def bounce_update(
    agent: UAVAgent,
    width: float,
    height: float,
    min_altitude: float,
    max_altitude: float,
) -> None:
    agent.x += agent.vx
    agent.y += agent.vy
    agent.z += agent.vz

    if agent.x < 0 or agent.x > width:
        agent.vx *= -1
        agent.x = min(max(agent.x, 0.0), width)
    if agent.y < 0 or agent.y > height:
        agent.vy *= -1
        agent.y = min(max(agent.y, 0.0), height)
    if agent.z < min_altitude or agent.z > max_altitude:
        agent.vz *= -1
        agent.z = min(max(agent.z, min_altitude), max_altitude)


def ap_distance(agent: UAVAgent, ap: APNode) -> float:
    return euclidean_distance_3d(agent.x, agent.y, agent.z, ap.x, ap.y, ap.z)
