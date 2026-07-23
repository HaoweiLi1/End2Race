-- Portable-report source views over the reviewed analysis snapshots.
-- The Python notebook creates these CSV files directly from the repository JSON,
-- JSONL, checkpoint, and NPZ artifacts before the report is packaged.
CREATE OR REPLACE VIEW control_audit AS
SELECT * FROM read_csv_auto('data/control_audit.csv', header = true);

CREATE OR REPLACE VIEW training_metrics AS
SELECT * FROM read_csv_auto('data/training_metrics.csv', header = true);

CREATE OR REPLACE VIEW training_summary AS
SELECT * FROM read_csv_auto('data/training_summary.csv', header = true);

CREATE OR REPLACE VIEW eval_summary AS
SELECT * FROM read_csv_auto('data/eval_summary.csv', header = true);

CREATE OR REPLACE VIEW eval_rollup AS
SELECT * FROM read_csv_auto('data/eval_rollup.csv', header = true);

CREATE OR REPLACE VIEW actor_parameter_deltas AS
SELECT * FROM read_csv_auto('data/actor_parameter_deltas.csv', header = true);

CREATE OR REPLACE VIEW scenario_slices AS
SELECT * FROM read_csv_auto('data/scenario_slices.csv', header = true);
