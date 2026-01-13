"""
Kernel Coverage Tracking for Genesis.

This module tracks which Taichi kernels are executed during tests.
Since standard Python coverage tools cannot instrument JIT-compiled kernel code,
this provides an alternative: tracking which kernels were called (function-level coverage).

Usage:
    # In conftest.py or test setup:
    from tests.coverage.kernel_coverage import KernelCoverageTracker

    tracker = KernelCoverageTracker()
    tracker.enable()

    # Run tests...

    tracker.collect()
    tracker.save_report("kernel_coverage.json")
    tracker.print_summary()
"""

import ast
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional


class KernelCoverageTracker:
    """Tracks execution of Taichi kernels during tests."""

    def __init__(self, genesis_root: Optional[Path] = None):
        """
        Initialize kernel coverage tracker.

        Args:
            genesis_root: Root directory of genesis package. Auto-detected if None.
        """
        if genesis_root is None:
            genesis_root = Path(__file__).parent.parent.parent / "genesis"
        self.genesis_root = Path(genesis_root)

        self._defined_kernels: dict[str, list[str]] = {}  # file -> [kernel_names]
        self._executed_kernels: set[str] = set()
        self._profiler_enabled = False
        self._ti = None

    def discover_kernels(self) -> dict[str, list[str]]:
        """
        Discover all @ti.kernel decorated functions in the codebase.

        Returns:
            Dict mapping file paths to list of kernel function names.
        """
        self._defined_kernels = {}

        for py_file in self.genesis_root.rglob("*.py"):
            # Skip external/vendored code
            if "ext/" in str(py_file):
                continue

            kernels = self._extract_kernels_from_file(py_file)
            if kernels:
                rel_path = str(py_file.relative_to(self.genesis_root.parent))
                self._defined_kernels[rel_path] = kernels

        return self._defined_kernels

    def _extract_kernels_from_file(self, filepath: Path) -> list[str]:
        """Extract kernel function names from a Python file using AST parsing."""
        kernels = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    for decorator in node.decorator_list:
                        # Handle @ti.kernel
                        if isinstance(decorator, ast.Attribute):
                            if decorator.attr == "kernel":
                                kernels.append(node.name)
                        # Handle @kernel (after from gstaichi import kernel)
                        elif isinstance(decorator, ast.Name):
                            if decorator.id == "kernel":
                                kernels.append(node.name)
                        # Handle @ti.kernel() with parentheses
                        elif isinstance(decorator, ast.Call):
                            if isinstance(decorator.func, ast.Attribute):
                                if decorator.func.attr == "kernel":
                                    kernels.append(node.name)
        except (SyntaxError, UnicodeDecodeError):
            pass

        return kernels

    def enable(self) -> bool:
        """
        Enable kernel profiling to track executed kernels.

        Returns:
            True if profiling was enabled successfully.
        """
        try:
            import gstaichi as ti

            self._ti = ti

            # Check if already initialized
            if hasattr(ti, "cfg") and ti.cfg is not None:
                # Taichi is initialized, enable profiler
                # Note: get_default_kernel_profiler is in ti.profiler.kernel_profiler
                ti.profiler.kernel_profiler.get_default_kernel_profiler().set_kernel_profiler_mode(True)
                self._profiler_enabled = True
                return True
            else:
                # Will need to enable after gs.init()
                self._profiler_enabled = False
                return False
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not enable kernel profiler: {e}")
            return False

    def enable_after_init(self):
        """Call this after gs.init() to enable profiling."""
        if self._ti is not None:
            try:
                self._ti.profiler.kernel_profiler.get_default_kernel_profiler().set_kernel_profiler_mode(True)
                self._profiler_enabled = True
            except Exception as e:
                print(f"Warning: Could not enable kernel profiler after init: {e}")

    def collect(self) -> set[str]:
        """
        Collect executed kernel names from the profiler.

        Returns:
            Set of kernel names that were executed.
        """
        if not self._profiler_enabled or self._ti is None:
            return self._executed_kernels

        try:
            profiler = self._ti.profiler.kernel_profiler.get_default_kernel_profiler()
            profiler._update_records()

            for record in profiler._traced_records:
                # Kernel names may have suffixes like "_c78_0_kernel"
                # Extract the base name
                name = record.name
                # Remove common suffixes added by Taichi
                name = re.sub(r"_c\d+_\d+.*", "", name)
                self._executed_kernels.add(name)
                # Also keep original for matching
                self._executed_kernels.add(record.name)
        except Exception as e:
            print(f"Warning: Could not collect kernel profiler data: {e}")

        return self._executed_kernels

    def get_coverage_stats(self) -> dict:
        """
        Calculate kernel coverage statistics.

        Returns:
            Dict with coverage statistics.
        """
        if not self._defined_kernels:
            self.discover_kernels()

        self.collect()

        total_kernels = sum(len(k) for k in self._defined_kernels.values())
        all_kernel_names = set()
        for kernels in self._defined_kernels.values():
            all_kernel_names.update(kernels)

        # Match executed kernels to defined kernels
        covered_kernels = set()
        for defined in all_kernel_names:
            for executed in self._executed_kernels:
                if defined in executed or executed in defined:
                    covered_kernels.add(defined)
                    break

        covered_count = len(covered_kernels)
        coverage_pct = (covered_count / total_kernels * 100) if total_kernels > 0 else 0

        # Per-file breakdown
        file_coverage = {}
        for filepath, kernels in self._defined_kernels.items():
            file_covered = [k for k in kernels if k in covered_kernels]
            file_coverage[filepath] = {
                "total": len(kernels),
                "covered": len(file_covered),
                "kernels": kernels,
                "covered_kernels": file_covered,
                "missing_kernels": [k for k in kernels if k not in covered_kernels],
            }

        return {
            "total_kernels": total_kernels,
            "covered_kernels": covered_count,
            "coverage_percent": round(coverage_pct, 2),
            "executed_kernel_names": list(self._executed_kernels),
            "file_coverage": file_coverage,
        }

    def print_summary(self):
        """Print a summary of kernel coverage."""
        stats = self.get_coverage_stats()

        print("\n" + "=" * 70)
        print("KERNEL COVERAGE SUMMARY")
        print("=" * 70)
        print(f"Total kernels defined:  {stats['total_kernels']}")
        print(f"Kernels executed:       {stats['covered_kernels']}")
        print(f"Coverage:               {stats['coverage_percent']:.1f}%")
        print("-" * 70)

        # Show files with lowest coverage
        file_stats = []
        for filepath, data in stats["file_coverage"].items():
            if data["total"] > 0:
                pct = data["covered"] / data["total"] * 100
                file_stats.append((filepath, data["covered"], data["total"], pct))

        file_stats.sort(key=lambda x: (x[3], x[2]))  # Sort by coverage %, then total

        print("\nFiles with lowest coverage:")
        for filepath, covered, total, pct in file_stats[:10]:
            print(f"  {pct:5.1f}% ({covered:3d}/{total:3d}) {filepath}")

        print("=" * 70)

    def save_report(self, filepath: str):
        """Save detailed coverage report to JSON file."""
        stats = self.get_coverage_stats()
        with open(filepath, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Kernel coverage report saved to: {filepath}")

    def clear(self):
        """Clear collected execution data."""
        self._executed_kernels.clear()
        if self._profiler_enabled and self._ti is not None:
            try:
                self._ti.profiler.clear_kernel_profiler_info()
            except Exception:
                pass


# Global tracker instance for pytest hooks
_global_tracker: Optional[KernelCoverageTracker] = None


def get_tracker() -> KernelCoverageTracker:
    """Get or create the global kernel coverage tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = KernelCoverageTracker()
    return _global_tracker
