"""Unit tests for the HF publisher.

All tests in this module are PR-CI-safe: they never hit the real HF Hub.
The :class:`huggingface_hub.HfApi` is patched out with a recording mock and
network-error paths are simulated by raising ``HfHubHTTPError`` from the mock.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from huggingface_hub.utils import HfHubHTTPError

from inferencebench.envelope import Envelope
from inferencebench_hf_publisher import (
    HfPublishError,
    HfPublishResult,
    HfRateLimitError,
    HfRepoCollisionError,
    compute_repo_id,
    publish_envelope_to_hf,
    render_envelope_readme,
    slugify,
)


# --------------------------------------------------------------------------- #
# Slugification + repo id                                                     #
# --------------------------------------------------------------------------- #
class TestSlugify:
    def test_lowercases(self) -> None:
        assert slugify("Meta-Llama/Llama-4") == "meta-llama-llama-4"

    def test_collapses_runs(self) -> None:
        assert slugify("a   b___c") == "a-b-c"

    def test_strips_edges(self) -> None:
        assert slugify("---hello---") == "hello"

    def test_empty_yields_unknown(self) -> None:
        assert slugify("") == "unknown"

    def test_only_invalid_yields_unknown(self) -> None:
        assert slugify("///---") == "unknown"

    def test_keeps_digits(self) -> None:
        assert slugify("Llama-4-405B") == "llama-4-405b"


class TestComputeRepoId:
    def test_default_org(self, envelope: Envelope) -> None:
        rid = compute_repo_id(envelope)
        assert rid.startswith("Yobitel/")
        # model-slug __ suite-slug __ run-hash(12)
        _, _, tail = rid.partition("Yobitel/")
        parts = tail.split("__")
        assert len(parts) == 3
        assert parts[0] == "meta-llama-llama-4-maverick"
        assert parts[1] == "llm-inference"
        assert len(parts[2]) == 12
        # run_id starts with 0193456789ab... → first 12 hex chars after stripping `-`
        assert parts[2] == "0193456789ab"

    def test_custom_org(self, envelope: Envelope) -> None:
        rid = compute_repo_id(envelope, org="staging-org")
        assert rid.startswith("staging-org/")


# --------------------------------------------------------------------------- #
# README rendering                                                            #
# --------------------------------------------------------------------------- #
class TestReadmeRendering:
    def test_contains_all_sections(self, envelope: Envelope) -> None:
        md = render_envelope_readme(envelope)
        assert "# meta-llama/Llama-4-Maverick on llm.inference" in md
        assert "## Headline metrics" in md
        assert "## Run configuration" in md
        assert "## Verification" in md
        assert "## Methodology" in md
        assert "## Citation" in md

    def test_has_yaml_frontmatter(self, envelope: Envelope) -> None:
        md = render_envelope_readme(envelope)
        # Frontmatter must be the very first thing in the file and closed.
        assert md.startswith("---\n")
        # Two fence markers must be present.
        assert md.count("\n---\n") >= 1
        assert "inferencebench:" in md
        assert "envelope_version: v1" in md
        assert "suite_id: llm.inference" in md
        assert "fingerprint_sha256:" in md
        assert "license: cc-by-4.0" in md

    def test_signature_verified_false_when_unsigned(self, envelope: Envelope) -> None:
        md = render_envelope_readme(envelope)
        assert "signature_verified: false" in md

    def test_signature_verified_true_when_signed(self, signed_envelope: Envelope) -> None:
        md = render_envelope_readme(signed_envelope)
        assert "signature_verified: true" in md
        assert "rekor_log_index: 987654" in md

    def test_metric_table_includes_humanized_label(self, envelope: Envelope) -> None:
        md = render_envelope_readme(envelope)
        assert "| TTFT P50 | 142 | ms |" in md
        assert "| Throughput |" in md
        assert "tok/s" in md

    def test_citation_uses_run_hash(self, envelope: Envelope) -> None:
        md = render_envelope_readme(envelope)
        assert "@misc{inferencebench_0193456789ab" in md

    def test_no_quantization_falls_back_cleanly(self, envelope: Envelope) -> None:
        # Build a near-copy without quantization to exercise the n/a branch.
        data = envelope.model_dump()
        data.pop("quantization", None)
        rebuilt = Envelope(**data)
        md = render_envelope_readme(rebuilt)
        assert "**Quantization**: n/a" in md

    def test_hardware_class_in_frontmatter(self, envelope: Envelope) -> None:
        md = render_envelope_readme(envelope)
        assert "hardware_class: h100" in md
        assert "- h100" in md  # also in tags list


# --------------------------------------------------------------------------- #
# Dry-run mode                                                                #
# --------------------------------------------------------------------------- #
class TestDryRun:
    def test_returns_planned_result_without_hitting_hub(self, envelope: Envelope) -> None:
        with mock.patch(
            "inferencebench_hf_publisher.publish.HfApi",
        ) as hf_api_cls:
            result = publish_envelope_to_hf(envelope, dry_run=True)
        hf_api_cls.assert_not_called()
        assert isinstance(result, HfPublishResult)
        assert result.repo_id == compute_repo_id(envelope)
        assert result.url == f"https://huggingface.co/datasets/{result.repo_id}"
        assert result.files_uploaded == []
        assert result.verified is False

    def test_dry_run_respects_org_override(self, envelope: Envelope) -> None:
        result = publish_envelope_to_hf(envelope, org="yobitel-bench-staging", dry_run=True)
        assert result.repo_id.startswith("yobitel-bench-staging/")


# --------------------------------------------------------------------------- #
# Happy-path with mocked HfApi                                                #
# --------------------------------------------------------------------------- #
class TestPublishHappyPath:
    def test_uploads_envelope_and_readme(self, envelope: Envelope) -> None:
        with mock.patch("inferencebench_hf_publisher.publish.HfApi") as hf_api_cls:
            mock_api = mock.MagicMock()
            hf_api_cls.return_value = mock_api
            result = publish_envelope_to_hf(envelope, hf_token="hf_test")

        hf_api_cls.assert_called_once_with(token="hf_test")
        mock_api.create_repo.assert_called_once()
        kwargs = mock_api.create_repo.call_args.kwargs
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["exist_ok"] is False
        assert kwargs["private"] is False
        # Two uploads minimum: envelope.json + README.md
        uploaded_paths = [
            call.kwargs["path_in_repo"] for call in mock_api.upload_file.call_args_list
        ]
        assert uploaded_paths == ["envelope.json", "README.md"]
        assert result.files_uploaded == ["envelope.json", "README.md"]
        assert result.verified is True

    def test_uploads_optional_traces_when_provided(
        self,
        envelope: Envelope,
        tmp_path: Path,
    ) -> None:
        traces = tmp_path / "traces.parquet"
        traces.write_bytes(b"PAR1...")  # dummy bytes; we mock the upload

        with mock.patch("inferencebench_hf_publisher.publish.HfApi") as hf_api_cls:
            mock_api = mock.MagicMock()
            hf_api_cls.return_value = mock_api
            result = publish_envelope_to_hf(envelope, raw_traces_path=traces)

        uploaded_paths = [
            call.kwargs["path_in_repo"] for call in mock_api.upload_file.call_args_list
        ]
        assert uploaded_paths == ["envelope.json", "traces.parquet", "README.md"]
        assert result.files_uploaded == ["envelope.json", "traces.parquet", "README.md"]

    def test_missing_traces_file_raises(self, envelope: Envelope, tmp_path: Path) -> None:
        with mock.patch("inferencebench_hf_publisher.publish.HfApi"), pytest.raises(
            HfPublishError,
            match="raw_traces_path does not exist",
        ):
            publish_envelope_to_hf(envelope, raw_traces_path=tmp_path / "missing.parquet")


# --------------------------------------------------------------------------- #
# Error paths                                                                 #
# --------------------------------------------------------------------------- #
def _http_error(status: int) -> HfHubHTTPError:
    """Build an HfHubHTTPError whose underlying response has a status_code."""
    resp = mock.MagicMock()
    resp.status_code = status

    class _Err(HfHubHTTPError):
        def __init__(self) -> None:
            super().__init__(f"http {status}", response=resp)

    return _Err()


class TestErrorClassification:
    def test_collision_on_create_repo(self, envelope: Envelope) -> None:
        with mock.patch("inferencebench_hf_publisher.publish.HfApi") as hf_api_cls:
            mock_api = mock.MagicMock()
            mock_api.create_repo.side_effect = _http_error(409)
            hf_api_cls.return_value = mock_api
            with pytest.raises(HfRepoCollisionError):
                publish_envelope_to_hf(envelope)

    def test_rate_limit_on_create_repo(self, envelope: Envelope) -> None:
        with mock.patch("inferencebench_hf_publisher.publish.HfApi") as hf_api_cls:
            mock_api = mock.MagicMock()
            mock_api.create_repo.side_effect = _http_error(429)
            hf_api_cls.return_value = mock_api
            with pytest.raises(HfRateLimitError):
                publish_envelope_to_hf(envelope)

    def test_generic_http_error_wrapped(self, envelope: Envelope) -> None:
        with mock.patch("inferencebench_hf_publisher.publish.HfApi") as hf_api_cls:
            mock_api = mock.MagicMock()
            mock_api.create_repo.side_effect = _http_error(500)
            hf_api_cls.return_value = mock_api
            with pytest.raises(HfPublishError):
                publish_envelope_to_hf(envelope)

    def test_rate_limit_on_envelope_upload(self, envelope: Envelope) -> None:
        with mock.patch("inferencebench_hf_publisher.publish.HfApi") as hf_api_cls:
            mock_api = mock.MagicMock()

            def _upload(**kwargs: Any) -> None:
                if kwargs.get("path_in_repo") == "envelope.json":
                    raise _http_error(429)

            mock_api.upload_file.side_effect = _upload
            hf_api_cls.return_value = mock_api
            with pytest.raises(HfRateLimitError):
                publish_envelope_to_hf(envelope)


# --------------------------------------------------------------------------- #
# Dataset card / frontmatter shape                                            #
# --------------------------------------------------------------------------- #
class TestDatasetCardMetadata:
    def test_frontmatter_keys_present(self, envelope: Envelope) -> None:
        md = render_envelope_readme(envelope)
        head = md.split("\n---\n", 2)[0].lstrip("-").strip()
        for required in (
            "license: cc-by-4.0",
            "language:",
            "size_categories:",
            "task_categories:",
            "tags:",
            "inferencebench:",
            "envelope_version: v1",
            "suite_id: llm.inference",
            "suite_version: 1.0.0",
            "model: meta-llama/Llama-4-Maverick",
            "engine: vllm",
            "fingerprint_sha256:",
            "rekor_log_index:",
        ):
            assert required in head, f"missing frontmatter key: {required!r}"

    def test_frontmatter_tags_include_modality_and_hw_class(
        self,
        envelope: Envelope,
    ) -> None:
        md = render_envelope_readme(envelope)
        head = md.split("\n---\n", 2)[0]
        assert "- benchmark" in head
        assert "- inferencebench" in head
        assert "- llm" in head  # modality from suite_id prefix
        assert "- h100" in head  # hardware class
