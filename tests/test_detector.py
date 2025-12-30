"""Tests for the DangerousCommandDetector."""

import pytest
from diterm.detector import DangerousCommandDetector


def test_detector_initialization():
    """Test that detector initializes correctly."""
    detector = DangerousCommandDetector()
    assert detector is not None
    assert len(detector.danger_patterns) > 0


def test_dangerous_rm_rf_root():
    """Test detection of rm -rf /."""
    detector = DangerousCommandDetector()
    assert detector.is_dangerous("rm -rf /") is True


def test_dangerous_rm_rf_wildcard():
    """Test detection of rm -rf *."""
    detector = DangerousCommandDetector()
    assert detector.is_dangerous("rm -rf *") is True


def test_dangerous_dd_command():
    """Test detection of dangerous dd commands."""
    detector = DangerousCommandDetector()
    assert detector.is_dangerous("dd if=/dev/zero of=/dev/sda") is True


def test_dangerous_drop_database():
    """Test detection of DROP DATABASE."""
    detector = DangerousCommandDetector()
    assert detector.is_dangerous("DROP DATABASE production") is True


def test_safe_commands():
    """Test that safe commands are not flagged."""
    detector = DangerousCommandDetector()
    safe_commands = [
        "ls -la",
        "cd /home",
        "echo hello",
        "grep test file.txt",
        "rm file.txt",
    ]
    for cmd in safe_commands:
        assert detector.is_dangerous(cmd) is False


def test_risk_score_calculation():
    """Test risk score calculation."""
    detector = DangerousCommandDetector()
    
    # Safe command should have 0 risk
    assert detector.get_risk_score("ls -la") == 0
    
    # Dangerous command should have > 0 risk
    assert detector.get_risk_score("rm -rf /") > 0
