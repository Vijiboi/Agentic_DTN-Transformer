from __future__ import annotations

from env.config import SimulationConfig
from env.graph import CommunicationGraph
from env.models import APNode, LocalSummary


class QuantizedConsensusCoordinator:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def update(
        self,
        ap_lookup: dict[str, APNode],
        graph: CommunicationGraph,
        local_summaries: list[LocalSummary],
        slot: int,
    ) -> None:
        summary_by_ap = {summary.ap_id: summary for summary in local_summaries}

        for ap_id, ap in ap_lookup.items():
            summary = summary_by_ap.get(ap_id)
            if summary is None:
                continue
            if not summary.coordination_triggered:
                ap.coordination_state.consensus_drift = 0.0
                continue

            neighbor_ids = graph.adjacency.get(ap_id, [])
            neighbor_duals = [ap_lookup[neighbor].coordination_state.quantized_dual for neighbor in neighbor_ids]
            neighbor_mean = sum(neighbor_duals) / max(len(neighbor_duals), 1)

            pressure = (
                0.55 * summary.queue_size / max(ap.cpu_capacity, 1.0)
                + 0.25 * summary.uncertainty
                + 0.20 * (1.0 - summary.trust)
            )
            delta = pressure - neighbor_mean
            next_dual = max(0.0, ap.coordination_state.dual_price + self.config.consensus_step_size * delta)
            quantized_dual = self._quantize(next_dual)

            ap.coordination_state.consensus_drift = delta
            ap.coordination_state.dual_price = next_dual
            ap.coordination_state.coupling_penalty = quantized_dual

            if abs(quantized_dual - ap.coordination_state.quantized_dual) > self.config.consensus_epsilon:
                ap.coordination_state.quantized_dual = quantized_dual
                ap.coordination_state.last_broadcast_slot = slot

    def _quantize(self, value: float) -> float:
        quantum = max(self.config.consensus_quantum, 1e-6)
        return round(value / quantum) * quantum
