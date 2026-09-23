"""
Unit tests for components/build_serving_image.py.

Mocks google.cloud.storage.Client and cloudbuild_v1.CloudBuildClient
entirely -- no real GCS upload, no real Cloud Build job, no cost.

The component writes its build context to a hardcoded "/tmp/build_context"
(not parameterized, unlike train_data/eval_data paths elsewhere in this
pipeline) -- that's a real testability gap worth knowing about, and it's
why these tests clean that directory up before/after rather than using
pytest's tmp_path for it directly.
"""
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conftest import FakeOutput

BUILD_DIR = Path("/tmp/build_context")


def _cleanup_build_dir():
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR, ignore_errors=True)


@patch("google.cloud.devtools.cloudbuild_v1.CloudBuildClient")
@patch("google.cloud.storage.Client")
def test_generated_main_py_embeds_correct_feature_lists(mock_storage_cls, mock_cb_cls, tmp_path):
    from components.build_serving_image import build_serving_image
    from google.cloud.devtools import cloudbuild_v1

    _cleanup_build_dir()
    try:
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"fake-model-bytes")
        model_in = FakeOutput(model_path)

        mock_storage_cls.return_value = MagicMock()

        mock_cb_client = MagicMock()
        mock_cb_cls.return_value = mock_cb_client
        mock_operation = MagicMock()
        mock_operation.result.return_value.status = cloudbuild_v1.Build.Status.SUCCESS
        mock_cb_client.create_build.return_value = mock_operation

        custom_cat_cols = ["custom_cat_a", "custom_cat_b"]
        custom_num_cols = ["custom_num_a"]
        image_uri_path = tmp_path / "image_uri.txt"

        build_serving_image.python_func(
            model=model_in,
            project_id="test-project",
            region="europe-west1",
            repo_name="test-repo",
            image_tag="unit-test",
            staging_bucket="gs://test-bucket",
            image_uri=str(image_uri_path),
            cat_cols=custom_cat_cols,
            num_cols=custom_num_cols,
        )

        generated_main_py = (BUILD_DIR / "main.py").read_text()

        # The exact lists passed in must appear in the generated code...
        assert repr(custom_cat_cols) in generated_main_py
        assert repr(custom_num_cols) in generated_main_py

        # ...and the old hardcoded defaults must NOT have leaked in instead
        # (this is the actual regression this test guards against).
        assert "product_category" not in generated_main_py

        # model.joblib was copied into the build context
        assert (BUILD_DIR / "model.joblib").read_bytes() == b"fake-model-bytes"

        assert (BUILD_DIR / "Dockerfile").exists()
        assert (BUILD_DIR / "requirements.txt").exists()

        written_uri = image_uri_path.read_text()
        assert written_uri == (
            "europe-west1-docker.pkg.dev/test-project/test-repo/serve-return-model:unit-test"
        )
    finally:
        _cleanup_build_dir()


@patch("google.cloud.devtools.cloudbuild_v1.CloudBuildClient")
@patch("google.cloud.storage.Client")
def test_raises_when_cloud_build_fails(mock_storage_cls, mock_cb_cls, tmp_path):
    """If Cloud Build reports a non-SUCCESS status, the component must raise
    rather than silently writing out an image_uri for an image that doesn't
    actually exist -- deploy_model would otherwise happily try to deploy it."""
    from components.build_serving_image import build_serving_image
    from google.cloud.devtools import cloudbuild_v1

    _cleanup_build_dir()
    try:
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"fake-model-bytes")

        mock_storage_cls.return_value = MagicMock()

        mock_cb_client = MagicMock()
        mock_cb_cls.return_value = mock_cb_client
        mock_operation = MagicMock()
        mock_operation.result.return_value.status = cloudbuild_v1.Build.Status.FAILURE
        mock_cb_client.create_build.return_value = mock_operation

        with pytest.raises(RuntimeError):
            build_serving_image.python_func(
                model=FakeOutput(model_path),
                project_id="test-project",
                region="europe-west1",
                repo_name="test-repo",
                image_tag="unit-test",
                staging_bucket="gs://test-bucket",
                image_uri=str(tmp_path / "image_uri.txt"),
            )
    finally:
        _cleanup_build_dir()