from __future__ import annotations

from random import Random
from time import perf_counter

from env.config import SimulationConfig
from env.models import QuboProblem, SolveResult


class ClassicalQuboSolver:
    def __init__(self, config: SimulationConfig, backend: str = "auto"):
        self.config = config
        self.backend = backend

    def solve(self, problem: QuboProblem) -> SolveResult:
        if not problem.variables:
            return SolveResult(sample={}, energy=0.0, samples=[{}], solver_name="empty", solver_time=0.0)

        if self.backend == "exact":
            return self._solve_exact(problem)

        if self.backend == "annealing":
            return self._solve_with_annealing(problem)

        if self.backend == "auto" and len(problem.variables) <= max(self.config.small_qubo_exact_threshold, 0):
            return self._solve_exact(problem)

        if self.backend == "neal":
            try:
                return self._solve_with_neal(problem)
            except Exception:
                pass

        if self.backend == "dimod":
            try:
                return self._solve_with_dimod(problem)
            except Exception:
                pass

        if self.backend == "auto":
            for method in (self._solve_with_neal, self._solve_with_dimod):
                try:
                    return method(problem)
                except Exception:
                    continue

        return self._solve_with_annealing(problem)

    def _solve_exact(self, problem: QuboProblem) -> SolveResult:
        start = perf_counter()
        best_sample: dict[tuple[str, str], int] | None = None
        best_energy = float("inf")
        samples: list[dict[tuple[str, str], int]] = []

        for mask in range(1 << len(problem.variables)):
            sample = {
                variable: (mask >> index) & 1
                for index, variable in enumerate(problem.variables)
            }
            energy = self._energy(problem, sample)
            samples.append(sample)
            if energy < best_energy:
                best_energy = energy
                best_sample = sample

        end = perf_counter()
        return SolveResult(
            sample=best_sample or {},
            energy=best_energy,
            samples=samples,
            solver_name="exact_enumeration",
            solver_time=end - start,
        )

    def _solve_with_annealing(self, problem: QuboProblem) -> SolveResult:
        rng = Random(self.config.seed + problem.slot + len(problem.variables))
        start = perf_counter()
        best_sample: dict[tuple[str, str], int] | None = None
        best_energy = float("inf")
        samples: list[dict[tuple[str, str], int]] = []

        for _ in range(self.config.anneal_reads):
            sample = {variable: rng.randint(0, 1) for variable in problem.variables}
            current_energy = self._energy(problem, sample)
            temperature = 2.5

            for sweep in range(self.config.anneal_sweeps):
                for variable in problem.variables:
                    trial = dict(sample)
                    trial[variable] = 1 - trial[variable]
                    candidate_energy = self._energy(problem, trial)
                    delta = candidate_energy - current_energy
                    accept = delta <= 0 or rng.random() < pow(2.718281828, -delta / max(temperature, 1e-9))
                    if accept:
                        sample = trial
                        current_energy = candidate_energy
                temperature *= 0.92 + (sweep / max(self.config.anneal_sweeps, 1)) * 0.01

            samples.append(sample)
            if current_energy < best_energy:
                best_energy = current_energy
                best_sample = sample

        end = perf_counter()
        return SolveResult(
            sample=best_sample or {},
            energy=best_energy,
            samples=samples,
            solver_name="classical_annealing",
            solver_time=end - start,
        )

    def _solve_with_neal(self, problem: QuboProblem) -> SolveResult:
        import dimod
        import neal

        start = perf_counter()
        bqm = dimod.BinaryQuadraticModel.from_qubo(self._build_qubo_dict(problem))
        sampler = neal.SimulatedAnnealingSampler()
        response = sampler.sample(bqm, num_reads=self.config.anneal_reads, num_sweeps=self.config.anneal_sweeps, seed=self.config.seed)
        first = response.first
        end = perf_counter()
        sample = {key: int(value) for key, value in first.sample.items()}
        samples = [{key: int(value) for key, value in row.sample.items()} for row in response.data(fields=["sample"])]
        return SolveResult(
            sample=sample,
            energy=float(first.energy),
            samples=samples,
            solver_name="neal_simulated_annealing",
            solver_time=end - start,
        )

    def _solve_with_dimod(self, problem: QuboProblem) -> SolveResult:
        import dimod

        start = perf_counter()
        bqm = dimod.BinaryQuadraticModel.from_qubo(self._build_qubo_dict(problem))
        sampler = dimod.SimulatedAnnealingSampler()
        response = sampler.sample(bqm, num_reads=self.config.anneal_reads, num_sweeps=self.config.anneal_sweeps)
        first = response.first
        end = perf_counter()
        sample = {key: int(value) for key, value in first.sample.items()}
        samples = [{key: int(value) for key, value in row.sample.items()} for row in response.data(fields=["sample"])]
        return SolveResult(
            sample=sample,
            energy=float(first.energy),
            samples=samples,
            solver_name="dimod_simulated_annealing",
            solver_time=end - start,
        )

    @staticmethod
    def _build_qubo_dict(problem: QuboProblem) -> dict[tuple[tuple[str, str], tuple[str, str]], float]:
        qubo_dict: dict[tuple[tuple[str, str], tuple[str, str]], float] = {}
        for variable, value in problem.linear.items():
            qubo_dict[(variable, variable)] = value
        for pair, value in problem.quadratic.items():
            qubo_dict[pair] = value
        return qubo_dict

    @staticmethod
    def _energy(problem: QuboProblem, sample: dict[tuple[str, str], int]) -> float:
        energy = 0.0
        for variable, coefficient in problem.linear.items():
            energy += coefficient * sample[variable]
        for (left, right), coefficient in problem.quadratic.items():
            energy += coefficient * sample[left] * sample[right]
        return energy
