"""Regression tests for config.schema_from_dict's handling of malformed
schema documents. This exact bug class ('float' object is not iterable,
from iterating a field that should be a list but arrived as a bare scalar)
has recurred multiple times across applies_when, classes/ranges/seg_classes,
and finally the top-level measurements field itself — each field that gets
iterated over needs an isinstance guard, or a malformed Supabase row crashes
retraining with an opaque TypeError instead of a clear, actionable message.
"""
from __future__ import annotations

import pytest

from config import schema_from_dict

VALID_MEASUREMENT = {
    "key": "condition",
    "label": "Condition",
    "type": "classification",
    "background_class": "Background",
    "classes": [{"name": "Background"}, {"name": "Healthy"}],
}


def test_schema_from_dict_rejects_non_list_measurements():
    with pytest.raises(ValueError, match="measurements"):
        schema_from_dict({"measurements": 5.0})


def test_schema_from_dict_rejects_non_dict_measurement_entry():
    with pytest.raises(ValueError, match="object"):
        schema_from_dict({"measurements": [5.0]})


def test_schema_from_dict_tolerates_malformed_applies_when_classes_ranges_seg_classes():
    # These fields degrade to "empty" rather than crashing, matching the
    # earlier fixes for this same bug class (0f7107d, 6054e1b).
    doc = {
        "measurements": [
            {**VALID_MEASUREMENT, "applies_when": 5.0},
            {
                "key": "health_score",
                "label": "Health score",
                "type": "regression",
                "ranges": 5.0,
            },
            {
                "key": "biofouling",
                "label": "Biofouling",
                "type": "segmentation",
                "seg_classes": 5.0,
            },
        ]
    }
    schema = schema_from_dict(doc)
    assert schema.find("condition").applies_when == []
    assert schema.find("health_score").ranges == []
    assert schema.find("biofouling").seg_classes == []


def test_schema_from_dict_accepts_well_formed_document():
    schema = schema_from_dict({"measurements": [VALID_MEASUREMENT]})
    assert [m.key for m in schema.measurements] == ["condition"]
