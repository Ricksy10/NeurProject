"""
Unit tests for inference utilities.

Tests:
- T045: Decision rule with calibrated threshold
- T046: Submission CSV format validation
- T047: Submission validation checks
- T048: Integration test for inference pipeline
"""

import pytest
import csv
import torch

from neur.infer import apply_calibrated_threshold, write_submission_csv, validate_submission


class TestCalibratedDecisionRule:
    """T045: Test decision rule with various probabilities and thresholds"""

    def test_chlorella_above_threshold(self):
        """If P(chlorella) >= threshold, predict chlorella (class 0)"""
        probs = torch.tensor([0.7, 0.1, 0.1, 0.05, 0.05])
        threshold = 0.6
        result = apply_calibrated_threshold(probs, threshold)
        assert result == 0, "Should predict chlorella when prob >= threshold"

    def test_chlorella_below_threshold(self):
        """If P(chlorella) < threshold, use argmax on other classes"""
        probs = torch.tensor([0.3, 0.1, 0.4, 0.1, 0.1])
        threshold = 0.5
        result = apply_calibrated_threshold(probs, threshold)
        # argmax([0.1, 0.4, 0.1, 0.1]) = 1, so predict class 1+1 = 2
        assert result == 2, "Should predict class with max prob among non-chlorella"

    def test_threshold_boundary(self):
        """Test exact threshold boundary"""
        probs = torch.tensor([0.5, 0.2, 0.2, 0.05, 0.05])
        threshold = 0.5
        result = apply_calibrated_threshold(probs, threshold)
        assert result == 0, "Should predict chlorella at exact threshold"

    def test_high_threshold(self):
        """Test with high threshold requiring strong confidence"""
        probs = torch.tensor([0.6, 0.15, 0.15, 0.05, 0.05])
        threshold = 0.8
        result = apply_calibrated_threshold(probs, threshold)
        assert result in [1, 2], "Should fall back to other classes with high threshold"

    def test_batch_predictions(self):
        """Test batch of predictions"""
        probs = torch.tensor(
            [
                [0.7, 0.1, 0.1, 0.05, 0.05],  # chlorella 0.7 >= 0.5 -> 0
                [
                    0.3,
                    0.5,
                    0.1,
                    0.05,
                    0.05,
                ],  # chlorella 0.3 < 0.5, argmax([0.5,0.1,0.05,0.05])=0 -> 1
                [0.4, 0.1, 0.3, 0.1, 0.1],  # chlorella 0.4 < 0.5, argmax([0.1,0.3,0.1,0.1])=1 -> 2
            ]
        )
        threshold = 0.5
        results = [apply_calibrated_threshold(p, threshold) for p in probs]
        assert results == [0, 1, 2], "Batch should process correctly"


class TestSubmissionFormat:
    """T046: Test submission CSV format matches requirements"""

    def test_csv_header(self, tmp_path):
        """Verify CSV has correct header"""
        predictions = [("subject_1", 0), ("subject_2", 1)]
        output_file = tmp_path / "submission.csv"

        write_submission_csv(predictions, str(output_file))

        with open(output_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["ID", "TARGET"], "Header must be ID,TARGET"

    def test_csv_content_format(self, tmp_path):
        """Verify each row has ID and TARGET"""
        predictions = [("subj_1", 0), ("subj_2", 3), ("subj_3", 2)]
        output_file = tmp_path / "submission.csv"

        write_submission_csv(predictions, str(output_file))

        with open(output_file, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            rows = list(reader)

        assert len(rows) == 3, "Should have 3 prediction rows"
        assert rows[0] == ["subj_1", "0"]
        assert rows[1] == ["subj_2", "3"]
        assert rows[2] == ["subj_3", "2"]

    def test_sorted_by_subject_id(self, tmp_path):
        """Verify rows are sorted by subject ID"""
        predictions = [("subj_3", 2), ("subj_1", 0), ("subj_2", 1)]
        output_file = tmp_path / "submission.csv"

        write_submission_csv(predictions, str(output_file))

        with open(output_file, "r") as f:
            reader = csv.reader(f)
            next(reader)
            rows = list(reader)

        ids = [row[0] for row in rows]
        assert ids == ["subj_1", "subj_2", "subj_3"], "Should be sorted by ID"

    def test_integer_targets(self, tmp_path):
        """Verify targets are integers 0-4"""
        predictions = [(f"subj_{i}", i % 5) for i in range(10)]
        output_file = tmp_path / "submission.csv"

        write_submission_csv(predictions, str(output_file))

        with open(output_file, "r") as f:
            reader = csv.reader(f)
            next(reader)
            targets = [int(row[1]) for row in reader]

        assert all(0 <= t <= 4 for t in targets), "All targets must be in [0,4]"


class TestSubmissionValidation:
    """T047: Test submission validation checks"""

    def test_valid_submission(self, tmp_path):
        """Valid submission should pass all checks"""
        submission_file = tmp_path / "submission.csv"
        with open(submission_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "TARGET"])
            writer.writerow(["subject_1", "0"])
            writer.writerow(["subject_2", "3"])

        # Should not raise exception
        validate_submission(str(submission_file), expected_count=2)

    def test_missing_header(self, tmp_path):
        """Should detect missing or wrong header"""
        submission_file = tmp_path / "submission.csv"
        with open(submission_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Subject", "Prediction"])  # Wrong header
            writer.writerow(["subject_1", "0"])

        with pytest.raises(ValueError, match="Header must be"):
            validate_submission(str(submission_file))

    def test_duplicate_ids(self, tmp_path):
        """Should detect duplicate subject IDs"""
        submission_file = tmp_path / "submission.csv"
        with open(submission_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "TARGET"])
            writer.writerow(["subject_1", "0"])
            writer.writerow(["subject_1", "2"])  # Duplicate

        with pytest.raises(ValueError, match="Duplicate"):
            validate_submission(str(submission_file))

    def test_out_of_range_targets(self, tmp_path):
        """Should detect targets outside [0,4]"""
        submission_file = tmp_path / "submission.csv"
        with open(submission_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "TARGET"])
            writer.writerow(["subject_1", "5"])  # Out of range

        with pytest.raises(ValueError, match="out of range"):
            validate_submission(str(submission_file))

    def test_wrong_row_count(self, tmp_path):
        """Should detect wrong number of predictions"""
        submission_file = tmp_path / "submission.csv"
        with open(submission_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "TARGET"])
            writer.writerow(["subject_1", "0"])

        with pytest.raises(ValueError, match="Expected 5 predictions"):
            validate_submission(str(submission_file), expected_count=5)


class TestInferencePipeline:
    """T048: Integration test for inference pipeline"""

    def test_end_to_end_inference(self, tmp_path, synthetic_test_subjects):
        """Test complete inference pipeline on synthetic data"""
        # This test would require more setup with actual model and data
        # For now, mark as placeholder that needs model fixtures
        pytest.skip("Requires model and checkpoint fixtures")

    def test_submission_generation(self, tmp_path):
        """Test that submission file is generated correctly"""
        # Mock predictions
        predictions = [(f"test_subject_{i}", i % 5) for i in range(10)]

        output_file = tmp_path / "submission.csv"
        write_submission_csv(predictions, str(output_file))

        # Validate
        validate_submission(str(output_file), expected_count=10)

        # Verify file exists and has content
        assert output_file.exists()
        with open(output_file, "r") as f:
            lines = f.readlines()
        assert len(lines) == 11  # Header + 10 predictions
