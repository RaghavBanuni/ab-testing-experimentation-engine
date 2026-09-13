"""End-to-end smoke tests: the documented commands must actually run."""

from __future__ import annotations

from ab_engine.cli import main
from ab_engine.make_dataset import make_experiment_frame


def test_power_command_for_a_continuous_metric(capsys):
    assert main(["power", "--sigma", "1.0", "--mde", "0.05"]) == 0
    assert "total units needed" in capsys.readouterr().out


def test_power_command_solves_for_mde(capsys):
    assert main(["power", "--sigma", "1.0", "--total-n", "50000"]) == 0
    assert "minimum detectable effect" in capsys.readouterr().out


def test_power_command_for_a_binary_metric(capsys):
    assert main(["power", "--baseline-rate", "0.1", "--relative-lift", "0.05"]) == 0
    assert "total units needed" in capsys.readouterr().out


def test_power_command_reports_missing_arguments(capsys):
    assert main(["power", "--sigma", "1.0"]) == 2


def test_bucket_command_reports_a_balanced_split(capsys):
    assert main(["bucket", "--n-units", "5000"]) == 0
    output = capsys.readouterr().out
    assert "SRM chi-square" in output
    assert "No sample ratio mismatch" in output


def test_simulate_command_runs_and_reports_every_method(capsys):
    exit_code = main(
        ["simulate", "--n-experiments", "40", "--n-per-arm", "200", "--n-looks", "4"]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    for method in ("fixed_final", "fixed_peeking", "msprt", "alpha_spending"):
        assert method in output
    assert "false-positive rate" in output


def test_analyse_command_runs_the_full_readout(tmp_path, capsys):
    frame = make_experiment_frame(n_units=4000, true_effect=0.3, seed=5)
    csv_path = tmp_path / "experiment.csv"
    frame.to_csv(csv_path, index=False)

    assert main(["analyse", str(csv_path)]) == 0
    output = capsys.readouterr().out
    assert "invalidation gates" in output
    assert "CUPED" in output
    assert "always-valid readout" in output
    assert "segments" in output
    assert "reversal detected" in output


def test_analyse_command_stops_when_the_srm_gate_fails(tmp_path, capsys):
    """A broken experiment must not get an effect reported at all."""
    import pandas as pd

    rows = [{"variant": "treatment", "metric": 1.0 + 0.001 * i} for i in range(5000)]
    rows += [{"variant": "control", "metric": 0.001 * i} for i in range(4000)]
    csv_path = tmp_path / "broken.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    assert main(["analyse", str(csv_path)]) == 1
    output = capsys.readouterr().out
    assert "Sample ratio mismatch" in output
    assert "Stopping" in output
    assert "always-valid readout" not in output


def test_generated_dataset_has_the_expected_columns():
    frame = make_experiment_frame(n_units=500)
    assert set(frame.columns) == {"unit_id", "variant", "metric", "pre_metric", "segment"}
    assert set(frame["variant"].unique()) <= {"control", "treatment"}
