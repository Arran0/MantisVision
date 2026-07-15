"""Integration test for the retrain orchestration's new (schema-driven,
column-annotation) data-materialization path: spins up a real local HTTP
server that mimics just enough of Supabase's PostgREST + Storage APIs, then
exercises scripts.export_schema.fetch_active_schema and
scripts.retrain_and_report.fetch_labeled_images/materialize_new_labels
against it for real (no mocking of urllib itself) — proving the exact HTTP
request/response shapes those functions build actually work, and that the
resulting on-disk layout is one src.data.dataset.AnnotatedDataset (Round 3)
can load back successfully.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
from PIL import Image

from config import schema_from_dict
from src.data.dataset import AnnotatedDataset

SCHEMA_DOC = {
    "species": [{"name": "Kappaphycus alvarezii", "slug": "Kappaphycus_alvarezii"}],
    "active_species_slug": "Kappaphycus_alvarezii",
    "health_moderate_min": 45.0,
    "health_healthy_min": 75.0,
    "measurements": [
        {
            "key": "condition",
            "label": "Condition",
            "type": "classification",
            "loss_weight": 1.0,
            "background_class": "Background",
            "classes": [{"name": "Background"}, {"name": "Healthy"}, {"name": "Disease"}],
        },
        {
            "key": "health_score",
            "label": "Health score",
            "type": "regression",
            "loss_weight": 1.0,
            "min": 0.0,
            "max": 100.0,
            "applies_when": {"key": "condition", "not_equals": "Background"},
        },
        {
            "key": "biofouling",
            "label": "Biofouling",
            "type": "segmentation",
            "loss_weight": 1.0,
            "seg_classes": [{"name": "background", "color": "#000000"}, {"name": "algae", "color": "#22c55e"}],
        },
    ],
}


def _png_bytes(size=16, fill=1) -> bytes:
    import io

    array = np.full((size, size), fill, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(array, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(size=16) -> bytes:
    import io

    array = np.random.default_rng(0).integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buf, format="JPEG")
    return buf.getvalue()


TRAINING_IMAGES_ROWS = [
    {"id": "img-bg", "storage_path": "Background/bg.jpg", "measurements": {"condition": "Background"}},
    {
        "id": "img-healthy-1",
        "storage_path": "Healthy/h1.jpg",
        "measurements": {"condition": "Healthy", "health_score": 88.0, "biofouling": "biofouling/h1.png"},
    },
    {"id": "img-healthy-2", "storage_path": "Healthy/h2.jpg", "measurements": {"condition": "Healthy", "health_score": 91.0}},
    {"id": "img-disease-1", "storage_path": "Disease/d1.jpg", "measurements": {"condition": "Disease"}},
]


class _MockSupabaseHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def _send(self, status: int, body: bytes, content_type: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/rest/v1/measurement_schema"):
            self._send(200, json.dumps([{"doc": SCHEMA_DOC}]).encode())
        elif self.path.startswith("/rest/v1/training_images"):
            self._send(200, json.dumps(TRAINING_IMAGES_ROWS).encode())
        elif self.path.startswith("/storage/v1/object/training-images/"):
            rel = self.path[len("/storage/v1/object/training-images/"):]
            self._send(200, _jpeg_bytes(), content_type="image/jpeg")
        elif self.path.startswith("/storage/v1/object/training-masks/"):
            self._send(200, _png_bytes(), content_type="image/png")
        else:
            self._send(404, b"{}")

    def do_PATCH(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self._send(200, b"[]")


@pytest.fixture()
def mock_supabase(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _MockSupabaseHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    monkeypatch.setenv("SUPABASE_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-key")
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    thread.join(timeout=5)


def test_fetch_active_schema_returns_the_mocked_doc(mock_supabase):
    from scripts.export_schema import fetch_active_schema

    doc = fetch_active_schema()
    assert doc == SCHEMA_DOC


def test_fetch_active_schema_falls_back_to_default_when_no_row(mock_supabase, monkeypatch):
    from scripts.export_schema import fetch_active_schema

    class _EmptyHandler(_MockSupabaseHandler):
        def do_GET(self):
            if self.path.startswith("/rest/v1/measurement_schema"):
                self._send(200, b"[]")
            else:
                super().do_GET()

    # Swap in a server that returns no schema rows.
    server = HTTPServer(("127.0.0.1", 0), _EmptyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("SUPABASE_URL", f"http://127.0.0.1:{server.server_address[1]}")
    try:
        doc = fetch_active_schema()
        assert doc["active_species_slug"] == "Kappaphycus_alvarezii"
        assert any(m["key"] == "seaweed_presence" for m in doc["measurements"])
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_materialize_new_labels_writes_a_dataset_the_annotated_dataset_can_load(mock_supabase, tmp_path, monkeypatch):
    # materialize_new_labels (like the rest of retrain_and_report.py) reads
    # paths off the module-level `config` singleton directly rather than
    # taking a Config argument, matching this script's existing style — so
    # redirect that singleton's dataset_root for the duration of the test
    # rather than constructing an unused separate Config.
    import config as config_module
    from scripts.retrain_and_report import fetch_labeled_images, materialize_new_labels

    monkeypatch.setattr(config_module.config, "dataset_root", tmp_path / "dataset")
    monkeypatch.setattr(config_module.config, "image_size", 16)
    cfg = config_module.config
    assert cfg.species_slug == "Kappaphycus_alvarezii"  # matches SCHEMA_DOC's active_species_slug

    schema = schema_from_dict(SCHEMA_DOC)
    images = fetch_labeled_images()
    assert len(images) == len(TRAINING_IMAGES_ROWS)

    raw_dir = tmp_path / "_incoming"
    total = materialize_new_labels(images, schema, raw_dir)
    assert total == len(TRAINING_IMAGES_ROWS)

    # Every split got the schema's split_class ratios applied to the same 4
    # rows; regardless of exactly how they landed, each split directory must
    # be loadable by AnnotatedDataset (Round 3) with no errors, and any split
    # containing the Healthy image with a mask must produce a real
    # (not placeholder) segmentation target.
    total_loaded = 0
    saw_real_mask = False
    for split_dir in (cfg.train_dir, cfg.val_dir, cfg.test_dir):
        if not (split_dir / "annotations.jsonl").exists():
            continue
        ds = AnnotatedDataset(split_dir, cfg, schema, train=False)
        for i in range(len(ds)):
            image, targets = ds[i]
            assert image.shape == (3, cfg.image_size, cfg.image_size)
            assert "condition_id" in targets
            assert "health_score" in targets
            assert "biofouling_seg" in targets
            if targets["biofouling_seg_mask"] == 1.0:
                saw_real_mask = True
        total_loaded += len(ds)

    assert total_loaded == len(TRAINING_IMAGES_ROWS)
    assert saw_real_mask, "the one row with a biofouling mask should have produced a real (unmasked) seg target"
