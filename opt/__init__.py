from __future__ import annotations
"""Local QUBO builders and solver backends[cite: 1202]."""
from .qubo_generator import LocalQuboBuilder
from .solver import ClassicalQuboSolver

__all__ = ["ClassicalQuboSolver", "LocalQuboBuilder"]