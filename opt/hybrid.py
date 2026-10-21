from __future__ import annotations

from time import perf_counter

from env.models import QuboProblem, SolveResult


class DWaveHybridSolver:
    """Phase 2 hook. The DTN logic stays classical and only this backend swaps in."""

    def solve(self, problem: QuboProblem) -> SolveResult:
        try:
            import dimod
            from dwave.system import LeapHybridSampler
        except Exception as exc:
            raise RuntimeError(
                "D-Wave dependencies are not installed. Install dimod and dwave-system before using the hybrid backend."
            ) from exc

        start = perf_counter()
        qubo_dict: dict[tuple[tuple[str, str], tuple[str, str]], float] = {}
        for variable, value in problem.linear.items():
            qubo_dict[(variable, variable)] = value
        for pair, value in problem.quadratic.items():
            qubo_dict[pair] = value

        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo_dict)
        sampler = LeapHybridSampler()
        response = sampler.sample(bqm)
        first = response.first
        end = perf_counter()
        sample = {key: int(value) for key, value in first.sample.items()}
        samples = [{key: int(value) for key, value in row.sample.items()} for row in response.data(fields=["sample"])]
        return SolveResult(
            sample=sample,
            energy=float(first.energy),
            samples=samples or [sample],
            solver_name="dwave_hybrid",
            solver_time=end - start,
        )
