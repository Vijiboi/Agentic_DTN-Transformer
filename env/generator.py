from __future__ import annotations

import math
import random
from collections import defaultdict
from heapq import nsmallest
from random import Random

from .config import SimulationConfig
from .graph import build_communication_graph
from .mobility import ap_distance, bounce_update
from .models import APNode, Observation, Task, TwinState, UAVAgent


class SaginEnvironment:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = Random(config.seed)
        self.aps = self._generate_aps()
        self.communication_graph = build_communication_graph(self.aps)
        self.uavs = self._generate_uavs()

    def _generate_aps(self) -> list[APNode]:
        """Generates SAGIN infrastructure nodes (BS, HAP, LEO) ."""
        aps: list[APNode] = []
        tier_counts = {
            "BS": self.config.num_bs,
            "HAP": self.config.num_haps,
            "LEO": self.config.num_leos,
        }
        for tier, count in tier_counts.items():
            for index in range(count):
                ap_id = f"{tier}_{index}"
                # Coordinate limits based on paper's simulation setup [cite: 517]
                coord_limit = 1000.0 if tier == "BS" else (2000.0 if tier == "HAP" else 5000.0)
                
                aps.append(
                    APNode(
                        ap_id=ap_id,
                        tier=tier,
                        x=self.rng.uniform(0.0, coord_limit),
                        y=self.rng.uniform(0.0, coord_limit),
                        z=self.config.altitude_by_tier[tier],
                        bandwidth=self.config.bandwidth_by_tier[tier],
                        cpu_capacity=self.config.cpu_capacity_by_tier[tier],
                        communication_budget=self.config.communication_budget_by_tier[tier],
                        power_budget=self.config.power_budget_by_tier[tier],
                        # Initialize trust in range [0.70, 0.95] [cite: 523]
                        trust=self.rng.uniform(self.config.ap_trust_init_min, self.config.ap_trust_init_max),
                        sync_threshold=self.config.sync_mismatch_threshold,
                        coord_threshold=self.config.coord_mismatch_threshold,
                        # Active DT as a control object [cite: 9, 18]
                        twin_state=TwinState(
                            fidelity=1.0, 
                            age=0, 
                            uncertainty=0.0, 
                            mismatch=0.0
                        ),
                    )
                )
        return aps

    def _generate_uavs(self) -> list[UAVAgent]:
        uavs: list[UAVAgent] = []
        for index in range(self.config.num_uavs):
            uavs.append(
                UAVAgent(
                    uav_id=f"UAV_{index}",
                    x=self.rng.uniform(0.0, self.config.area_width),
                    y=self.rng.uniform(0.0, self.config.area_height),
                    z=self.rng.uniform(self.config.uav_altitude_min, self.config.uav_altitude_max),
                    vx=self.rng.uniform(self.config.uav_velocity_xy_min, self.config.uav_velocity_xy_max),
                    vy=self.rng.uniform(self.config.uav_velocity_xy_min, self.config.uav_velocity_xy_max),
                    vz=self.rng.uniform(self.config.uav_velocity_z_min, self.config.uav_velocity_z_max),
                )
            )
        return uavs

    def step_mobility(self) -> None:
        """Updates UAV positions; mobility induces prediction uncertainty [cite: 151-152]."""
        for uav in self.uavs:
            bounce_update(
                uav,
                self.config.area_width,
                self.config.area_height,
                self.config.uav_altitude_min,
                self.config.uav_max_altitude,
            )

    def ap_by_id(self) -> dict[str, APNode]:
        return {ap.ap_id: ap for ap in self.aps}

    def candidate_aps_for_uav(self, uav: UAVAgent) -> list[str]:
        """Returns M_u: candidate infrastructure nodes for task offloading[cite: 11, 22]."""
        limit = min(self.config.candidate_ap_limit, len(self.aps))
        if limit >= len(self.aps):
            return [ap.ap_id for ap in sorted(self.aps, key=lambda ap: ap_distance(uav, ap))]

        selected: list[APNode] = []
        selected_ids: set[str] = set()
        for tier in ("BS", "HAP", "LEO"):
            tier_aps = [ap for ap in self.aps if ap.tier == tier]
            if tier_aps and len(selected) < limit:
                nearest = min(tier_aps, key=lambda ap: ap_distance(uav, ap))
                selected.append(nearest)
                selected_ids.add(nearest.ap_id)

        for ap in sorted(self.aps, key=lambda item: ap_distance(uav, item)):
            if len(selected) >= limit:
                break
            if ap.ap_id not in selected_ids:
                selected.append(ap)
                selected_ids.add(ap.ap_id)

        return [ap.ap_id for ap in selected]

    def serving_ap_for_uav(self, uav: UAVAgent) -> str:
        return self.candidate_aps_for_uav(uav)[0]

    def create_tasks_for_slot(self, slot: int) -> list[Task]:
        """Generates emergency tasks with heterogeneous mission requirements [cite: 53-56, 111]."""
        tasks: list[Task] = []
        for uav in self.uavs:
            if self.rng.random() > self.config.task_arrival_probability:
                continue
            
            # Map to paper classes: control, imagery, alert 
            task_class = self.rng.choice(["control", "imagery", "alert"])
            candidate_ap_ids = self.candidate_aps_for_uav(uav)
            
            tasks.append(
                Task(
                    task_id=f"T{slot}_{uav.uav_id}",
                    source_uav=uav.uav_id,
                    owner_ap_id=candidate_ap_ids[0],
                    x=uav.x,
                    y=uav.y,
                    z=uav.z,
                    L_u=self.rng.uniform(0.5, 2.0), # Input size in Mbits 
                    D_u=1000.0,                     # CPU cycles per bit 
                    omega_u=self.rng.uniform(0.8, 1.4), # Mission-priority weight [cite: 56]
                    psi_u=self.rng.uniform(0.7, 1.3),   # Mission-risk sensitivity [cite: 56]
                    xi_u=task_class,
                    bandwidth_demand=self.rng.uniform(1.0, 5.0),
                    cpu_demand=self.rng.uniform(2.0, 8.0),
                    power_demand=self.rng.uniform(0.8, 3.5),
                    A_u_t=candidate_ap_ids,
                    arrival_slot=slot,
                    AoI=self.rng.randint(self.config.task_aoi_min, self.config.task_aoi_max)
                )
            )
        return tasks

    def group_tasks_by_owner(self, tasks: list[Task]) -> dict[str, list[Task]]:
        grouped: dict[str, list[Task]] = defaultdict(list)
        for task in tasks:
            grouped[task.owner_ap_id].append(task)
        return grouped

    def build_observation(
        self,
        ap: APNode,
        slot: int,
        queue: list[Task],
        total_tasks: int,
    ) -> Observation:
        """Builds s_m(t): the physical observation of the node state [cite: 114-115]."""
        queue_size = len(queue)
        queued_cpu = sum(task.cpu_demand for task in queue)
        queued_bandwidth = sum(task.bandwidth_demand for task in queue)
        
        # Observed load, reliability, and mission-pressure summaries [cite: 115]
        load_ratio = self._clip_ratio((ap.current_cpu_load + queued_cpu) / max(ap.cpu_capacity, 1.0))
        bandwidth_ratio = self._clip_ratio(queued_bandwidth / max(ap.communication_budget, 1.0))
        cpu_ratio = self._clip_ratio(queued_cpu / max(ap.cpu_capacity, 1.0))
        
        overlap = 0.0
        if total_tasks:
            overlap = sum(len(task.A_u_t) > 1 for task in queue) / max(queue_size, 1)

        mobility_pressure = 0.0
        if queue:
            uav_by_id = {uav.uav_id: uav for uav in self.uavs}
            speeds = []
            for task in queue:
                uav = uav_by_id.get(task.source_uav)
                if uav is not None:
                    speeds.append(math.sqrt(uav.vx**2 + uav.vy**2 + uav.vz**2))
            if speeds:
                mobility_pressure = self._clip_ratio(sum(speeds) / len(speeds) / 30.0)
            
        return Observation(
            ap_id=ap.ap_id,
            slot=slot,
            load_ratio=load_ratio,
            bandwidth_ratio=bandwidth_ratio,
            cpu_ratio=cpu_ratio,
            queue_size=queue_size,
            candidate_overlap=overlap,
            mobility_pressure=mobility_pressure,
        )

    def _clip_ratio(self, value: float) -> float:
        clip = self.config.observation_ratio_clip
        if clip is None:
            return value
        return min(value, clip)
