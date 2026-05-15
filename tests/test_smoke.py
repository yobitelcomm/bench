"""Smoke test — proves pytest is wired correctly. Replace as real tests come in."""


def test_truth() -> None:
    """Verify pytest can run and asserts work."""
    assert True


def test_python_version_at_least_3_12() -> None:
    """We require Python 3.12+; fail fast if someone runs us on older Python."""
    import sys

    assert sys.version_info >= (3, 12), f"Requires Python 3.12+, got {sys.version_info}"
