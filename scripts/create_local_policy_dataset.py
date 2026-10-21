"""Create option-1 labels from an existing candidate-level CSV.

The output keeps every original feature and PSO reference column, but adds
``policy_selected``.  That label is the minimum observable per-task score:

    candidate_local_cost + coord_coupling_penalty

This matches the information available to the existing per-task Set
Transformer, unlike a global PSO assignment that also depends on other tasks
in the time slot.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


GROUP_COLUMNS = ("scenario_id", "slot", "task_id")
SCORE_COLUMNS = ("candidate_local_cost", "coord_coupling_penalty")
POLICY_COLUMNS = ("policy_selected", "policy_selected_destination_id", "policy_rule")
POLICY_RULE = "argmin(candidate_local_cost + coord_coupling_penalty)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add per-task observable-cost policy labels to a candidate CSV."
    )
    parser.add_argument("--input", type=Path, required=True, help="source candidate CSV")
    parser.add_argument(
        "--output", type=Path, default=Path("per_task_policy_dataset_1m.csv"),
        help="new CSV; source is never modified",
    )
    return parser.parse_args()


def group_key(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(row[column] for column in GROUP_COLUMNS)  # type: ignore[return-value]


def score(row: dict[str, str]) -> float:
    return sum(float(row[column]) for column in SCORE_COLUMNS)


def write_group(writer: csv.DictWriter, rows: list[dict[str, str]]) -> tuple[int, bool]:
    if not rows:
        return 0, True
    winner = min(rows, key=lambda row: (score(row), int(row["candidate_rank"])))
    destination = winner["destination_id"]
    valid = len(rows) == int(rows[0]["candidate_count"])
    for row in rows:
        row["policy_selected"] = str(int(row is winner))
        row["policy_selected_destination_id"] = destination
        row["policy_rule"] = POLICY_RULE
        writer.writerow(row)
    return len(rows), valid


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if input_path == output_path:
        raise SystemExit("--output must be a new file; the source CSV is preserved.")
    if not input_path.is_file():
        raise SystemExit(f"Input CSV not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = groups = invalid_groups = 0
    profile_counts: dict[str, int] = {}
    with input_path.open("r", newline="", encoding="utf-8") as source, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise SystemExit("Input CSV has no header.")
        missing = set(GROUP_COLUMNS + SCORE_COLUMNS + ("candidate_rank", "candidate_count", "destination_id")) - set(reader.fieldnames)
        if missing:
            raise SystemExit(f"Input CSV is missing required columns: {sorted(missing)}")
        fieldnames = list(reader.fieldnames)
        for column in POLICY_COLUMNS:
            if column not in fieldnames:
                fieldnames.append(column)
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()

        current_key: tuple[str, str, str] | None = None
        current_rows: list[dict[str, str]] = []
        for row in reader:
            key = group_key(row)
            if current_key is not None and key != current_key:
                written, valid = write_group(writer, current_rows)
                rows_written += written
                groups += 1
                invalid_groups += int(not valid)
                profile = current_rows[0]["scenario_profile"]
                profile_counts[profile] = profile_counts.get(profile, 0) + 1
                current_rows = []
            current_key = key
            current_rows.append(row)
        if current_rows:
            written, valid = write_group(writer, current_rows)
            rows_written += written
            groups += 1
            invalid_groups += int(not valid)
            profile = current_rows[0]["scenario_profile"]
            profile_counts[profile] = profile_counts.get(profile, 0) + 1

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({
        "source_dataset": str(input_path), "dataset_path": str(output_path),
        "rows": rows_written, "task_groups": groups, "invalid_candidate_groups": invalid_groups,
        "label_column": "policy_selected", "label_rule": POLICY_RULE,
        "profiles": profile_counts,
    }, indent=2), encoding="utf-8")
    print(f"Dataset saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Rows: {rows_written:,}; task groups: {groups:,}; invalid groups: {invalid_groups:,}")


if __name__ == "__main__":
    main()
