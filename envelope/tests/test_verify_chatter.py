"""Regression tests for _quiet_sigstore_chatter.

The helper exists to silence two specific kinds of sigstore-python noise during
a successful keyless verify:

1. ``Failed to load a trusted root key: unsupported key type:`` — written via
   ``sys.stderr.write`` while ``Verifier.production()`` loads the TUF root.
2. ``unsafe (no-op) verification policy used! no verification performed!`` —
   written via ``sys.stderr.write`` from inside ``verify_artifact`` when the
   UnsafeNoOp policy is used.
3. ``Key <hex> failed to verify root`` — logged at WARNING by python-tuf /
   securesystemslib during root-metadata key validation.

If sigstore-python ever starts emitting *new* warnings, those must surface to
the operator. The filter is a deny-list, not a wholesale stderr-swallow.
"""

from __future__ import annotations

import logging
import sys

from inferencebench.envelope.verify import _quiet_sigstore_chatter


def test_known_stderr_noise_is_filtered(capsys) -> None:
    with _quiet_sigstore_chatter():
        sys.stderr.write("Failed to load a trusted root key: unsupported key type: 7\n")
        sys.stderr.write("unsafe (no-op) verification policy used! no verification performed!\n")

    captured = capsys.readouterr()
    assert captured.err == ""


def test_unknown_stderr_passes_through(capsys) -> None:
    with _quiet_sigstore_chatter():
        sys.stderr.write("a real warning from sigstore-python in some future release\n")

    captured = capsys.readouterr()
    assert "a real warning from sigstore-python" in captured.err


def test_mixed_known_and_unknown(capsys) -> None:
    with _quiet_sigstore_chatter():
        sys.stderr.write("Failed to load a trusted root key: unsupported key type: 9\n")
        sys.stderr.write("genuinely surprising message\n")
        sys.stderr.write("unsafe (no-op) verification policy used! no verification performed!\n")

    captured = capsys.readouterr()
    assert "genuinely surprising message" in captured.err
    assert "Failed to load a trusted root key" not in captured.err
    assert "unsafe (no-op)" not in captured.err


def test_noisy_loggers_are_quieted_then_restored(capsys) -> None:
    tuf_logger = logging.getLogger("tuf")
    before = tuf_logger.level

    with _quiet_sigstore_chatter():
        assert tuf_logger.level == logging.ERROR
        tuf_logger.warning("Key abc123 failed to verify root")

    assert tuf_logger.level == before
    captured = capsys.readouterr()
    assert "failed to verify root" not in captured.err


def test_logger_levels_restored_on_exception() -> None:
    tuf_logger = logging.getLogger("tuf")
    before = tuf_logger.level

    class BoomError(Exception):
        pass

    try:
        with _quiet_sigstore_chatter():
            raise BoomError("simulated mid-verify failure")
    except BoomError:
        pass

    assert tuf_logger.level == before
