from __future__ import annotations

from dataclasses import dataclass

from .mobility import euclidean_distance_3d
from .models import APNode


@dataclass(slots=True)
class CommunicationGraph:
    adjacency: dict[str, list[str]]
    edges: list[tuple[str, str]]


def build_communication_graph(aps: list[APNode]) -> CommunicationGraph:
    adjacency: dict[str, set[str]] = {ap.ap_id: set() for ap in aps}
    edges: list[tuple[str, str]] = []
    by_tier: dict[str, list[APNode]] = {"BS": [],"UAV": [], "HAP": [], "LEO": []}
    for ap in aps:
        by_tier.setdefault(ap.tier, []).append(ap)

    for bs in by_tier.get("BS", []):
        if by_tier.get("HAP"):
            hap = _nearest(bs, by_tier["HAP"])
            _connect(adjacency, edges, bs.ap_id, hap.ap_id)
            _connect(adjacency, edges, hap.ap_id, bs.ap_id)

    for hap in by_tier.get("HAP", []):
        for peer in by_tier.get("HAP", []):
            if hap.ap_id != peer.ap_id:
                _connect(adjacency, edges, hap.ap_id, peer.ap_id)
        if by_tier.get("LEO"):
            leo = _nearest(hap, by_tier["LEO"])
            _connect(adjacency, edges, hap.ap_id, leo.ap_id)
            _connect(adjacency, edges, leo.ap_id, hap.ap_id)

    return CommunicationGraph(
        adjacency={node: sorted(neighbors) for node, neighbors in adjacency.items()},
        edges=edges,
    )


def _nearest(source: APNode, targets: list[APNode]) -> APNode:
    return min(
        targets,
        key=lambda target: euclidean_distance_3d(
            source.x,
            source.y,
            source.z,
            target.x,
            target.y,
            target.z,
        ),
    )


def _connect(
    adjacency: dict[str, set[str]],
    edges: list[tuple[str, str]],
    src: str,
    dst: str,
) -> None:
    if dst not in adjacency[src]:
        adjacency[src].add(dst)
        edges.append((src, dst))
