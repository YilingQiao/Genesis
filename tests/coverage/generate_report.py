#!/usr/bin/env python
"""
Generate combined coverage report for Genesis.

This script combines:
1. Python code coverage (from coverage.py/pytest-cov)
2. Kernel execution coverage (from GSTaichi profiler)

Usage:
    python tests/coverage/generate_report.py

    # Or with specific paths:
    python tests/coverage/generate_report.py --coverage-data .coverage_data \
        --kernel-data kernel_coverage.json --output coverage_report
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def combine_coverage_files(data_dir: Path) -> bool:
    """Combine parallel coverage files into one."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"Coverage data directory not found: {data_dir}")
        return False

    # Find all .coverage.* files (parallel coverage files)
    coverage_files = list(data_dir.glob(".coverage.*"))
    if coverage_files:
        print(f"Found {len(coverage_files)} parallel coverage files, combining...")
        # Run coverage combine from within the data directory
        # This ensures it finds all .coverage.* files
        result = subprocess.run(
            ["coverage", "combine"],
            cwd=str(data_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Warning: coverage combine failed: {result.stderr}")
        else:
            print(f"Successfully combined {len(coverage_files)} coverage files")

    # Verify combined file exists
    combined_file = data_dir / ".coverage"
    if combined_file.exists():
        print(f"Combined coverage file: {combined_file}")
        return True
    else:
        print(f"Warning: Combined coverage file not found at {combined_file}")
        return False


def generate_python_coverage_report(data_dir: Path, output_dir: Path) -> dict:
    """Generate Python coverage report and return statistics."""
    data_file = Path(data_dir) / ".coverage"
    if not data_file.exists():
        print(f"No Python coverage data found at {data_file}")
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate HTML report
    html_dir = output_dir / "python"
    subprocess.run(
        [
            "coverage",
            "html",
            f"--data-file={data_file}",
            f"--directory={html_dir}",
            "--title=Genesis Python Coverage",
        ],
        capture_output=True,
    )

    # Generate JSON report for parsing
    json_file = output_dir / "python_coverage.json"
    result = subprocess.run(
        ["coverage", "json", f"--data-file={data_file}", "-o", str(json_file)],
        capture_output=True,
        text=True,
    )

    if json_file.exists():
        with open(json_file) as f:
            data = json.load(f)
        return {
            "covered_lines": data.get("totals", {}).get("covered_lines", 0),
            "missing_lines": data.get("totals", {}).get("missing_lines", 0),
            "total_lines": data.get("totals", {}).get("num_statements", 0),
            "percent_covered": data.get("totals", {}).get("percent_covered", 0),
            "files": data.get("files", {}),
        }

    return {}


def generate_kernel_coverage_summary(kernel_data_file: Path) -> dict:
    """Load kernel coverage data."""
    if not Path(kernel_data_file).exists():
        print(f"No kernel coverage data found at {kernel_data_file}")
        return {}

    with open(kernel_data_file) as f:
        return json.load(f)


def generate_combined_html_report(python_stats: dict, kernel_stats: dict, output_dir: Path):
    """Generate a combined HTML summary report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    python_pct = python_stats.get("percent_covered", 0)
    kernel_pct = kernel_stats.get("coverage_percent", 0)

    # Calculate combined score (weighted average: 70% Python + 30% Kernel)
    # Always use the weighted formula so missing kernel coverage is reflected in the score
    combined_pct = python_pct * 0.7 + kernel_pct * 0.3

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Genesis Coverage Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1; min-width: 250px; }}
        .card h3 {{ margin-top: 0; color: #666; font-size: 14px; text-transform: uppercase; }}
        .percent {{ font-size: 48px; font-weight: bold; }}
        .percent.high {{ color: #4CAF50; }}
        .percent.medium {{ color: #FF9800; }}
        .percent.low {{ color: #F44336; }}
        .details {{ color: #888; font-size: 14px; margin-top: 10px; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f8f8; font-weight: 600; }}
        .bar {{ height: 8px; background: #eee; border-radius: 4px; overflow: hidden; }}
        .bar-fill {{ height: 100%; background: #4CAF50; transition: width 0.3s; }}
        .bar-fill.medium {{ background: #FF9800; }}
        .bar-fill.low {{ background: #F44336; }}
        a {{ color: #1976D2; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .note {{ background: #FFF3CD; border: 1px solid #FFEEBA; border-radius: 4px; padding: 15px; margin: 20px 0; }}
        .timestamp {{ color: #888; font-size: 12px; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Genesis Coverage Report</h1>

        <div class="summary">
            <div class="card">
                <h3>Combined Score</h3>
                <div class="percent {get_color_class(combined_pct)}">{combined_pct:.1f}%</div>
                <div class="details">Weighted: 70% Python + 30% Kernel</div>
            </div>
            <div class="card">
                <h3>Python Coverage</h3>
                <div class="percent {get_color_class(python_pct)}">{python_pct:.1f}%</div>
                <div class="details">{python_stats.get("covered_lines", 0):,} / {python_stats.get("total_lines", 0):,} lines</div>
            </div>
            <div class="card">
                <h3>Kernel Coverage</h3>
                <div class="percent {get_color_class(kernel_pct)}">{kernel_pct:.1f}%</div>
                <div class="details">{kernel_stats.get("covered_kernels", 0)} / {kernel_stats.get("total_kernels", 0)} kernels</div>
            </div>
        </div>

        <div class="note">
            <strong>Note:</strong> Kernel coverage measures which <code>@ti.kernel</code> functions were called during tests.
            Due to JIT compilation, line-level coverage inside kernels cannot be measured with standard Python tools.
            Python coverage excludes lines inside <code>@ti.kernel</code> and <code>@ti.func</code> decorated functions.
        </div>

        <h2>Quick Links</h2>
        <ul>
            <li><a href="python/index.html">Detailed Python Coverage Report</a></li>
            <li><a href="kernel_coverage.json">Kernel Coverage Data (JSON)</a></li>
        </ul>

        {generate_kernel_table(kernel_stats)}

        <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
</body>
</html>
"""

    index_file = output_dir / "index.html"
    with open(index_file, "w") as f:
        f.write(html)

    print(f"Combined report generated: {index_file}")

    # Copy kernel data to output
    if kernel_stats:
        with open(output_dir / "kernel_coverage.json", "w") as f:
            json.dump(kernel_stats, f, indent=2)


def get_color_class(percent: float) -> str:
    """Get CSS class based on coverage percentage."""
    if percent >= 80:
        return "high"
    elif percent >= 50:
        return "medium"
    return "low"


def generate_kernel_table(kernel_stats: dict) -> str:
    """Generate HTML table for kernel coverage by file."""
    if not kernel_stats.get("file_coverage"):
        return ""

    rows = []
    for filepath, data in sorted(
        kernel_stats["file_coverage"].items(),
        key=lambda x: (x[1]["covered"] / x[1]["total"] if x[1]["total"] > 0 else 0),
    ):
        if data["total"] == 0:
            continue
        pct = data["covered"] / data["total"] * 100
        color_class = get_color_class(pct)
        rows.append(
            f"""<tr>
            <td>{filepath}</td>
            <td>{data["covered"]} / {data["total"]}</td>
            <td>
                <div class="bar"><div class="bar-fill {color_class}" style="width: {pct}%"></div></div>
            </td>
            <td>{pct:.1f}%</td>
        </tr>"""
        )

    if not rows:
        return ""

    return f"""
        <h2>Kernel Coverage by File</h2>
        <table>
            <thead>
                <tr>
                    <th>File</th>
                    <th>Kernels</th>
                    <th>Coverage</th>
                    <th>%</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows[:50])}
            </tbody>
        </table>
        <p class="details">Showing top 50 files. See kernel_coverage.json for full data.</p>
    """


def main():
    parser = argparse.ArgumentParser(description="Generate Genesis coverage report")
    parser.add_argument(
        "--coverage-data",
        default=".coverage_data",
        help="Directory containing coverage.py data files",
    )
    parser.add_argument(
        "--kernel-data",
        default="kernel_coverage.json",
        help="Path to kernel coverage JSON file",
    )
    parser.add_argument(
        "--output",
        default="coverage_report",
        help="Output directory for reports",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Genesis Coverage Report Generator")
    print("=" * 60)

    # Combine parallel coverage files
    print("\n1. Processing Python coverage data...")
    combine_coverage_files(Path(args.coverage_data))

    # Generate Python report
    print("2. Generating Python coverage report...")
    python_stats = generate_python_coverage_report(Path(args.coverage_data), Path(args.output))

    # Load kernel coverage
    print("3. Loading kernel coverage data...")
    kernel_stats = generate_kernel_coverage_summary(Path(args.kernel_data))

    # Generate combined report
    print("4. Generating combined report...")
    generate_combined_html_report(python_stats, kernel_stats, Path(args.output))

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if python_stats:
        print(f"Python Coverage:  {python_stats.get('percent_covered', 0):.1f}%")
        print(f"  Lines covered:  {python_stats.get('covered_lines', 0):,} / {python_stats.get('total_lines', 0):,}")
    if kernel_stats:
        print(f"Kernel Coverage:  {kernel_stats.get('coverage_percent', 0):.1f}%")
        print(f"  Kernels covered: {kernel_stats.get('covered_kernels', 0)} / {kernel_stats.get('total_kernels', 0)}")
    print("=" * 60)
    print(f"\nOpen {args.output}/index.html to view the full report")


if __name__ == "__main__":
    main()
