"""Tests for the conceptual framework."""

from backend.ml.framework import (
    FRAMEWORK,
    classify_maturity,
    MaturityLevel,
    get_all_questions,
    get_question_to_dimension_map,
    maturity_description,
)


def test_framework_has_five_dimensions():
    assert len(FRAMEWORK) == 5


def test_dimension_weights_sum_to_one():
    total = sum(d.weight for d in FRAMEWORK)
    assert abs(total - 1.0) < 1e-9


def test_each_dimension_has_five_subdimensions():
    for d in FRAMEWORK:
        assert len(d.sub_dimensions) == 5, f"{d.id} has {len(d.sub_dimensions)}"


def test_total_25_questions():
    assert len(get_all_questions()) == 25


def test_question_ids_unique():
    ids = [q.id for q in get_all_questions()]
    assert len(ids) == len(set(ids))


def test_question_dimension_mapping_complete():
    mapping = get_question_to_dimension_map()
    assert len(mapping) == 25
    for qid, dim_id in mapping.items():
        assert dim_id in [d.id for d in FRAMEWORK]


def test_scale_labels_have_five_levels():
    for q in get_all_questions():
        assert set(q.scale_labels.keys()) == {1, 2, 3, 4, 5}


def test_maturity_classification_boundaries():
    assert classify_maturity(0.5) == MaturityLevel.INITIAL
    assert classify_maturity(1.0) == MaturityLevel.INITIAL
    assert classify_maturity(1.5) == MaturityLevel.EXPLORING
    assert classify_maturity(2.5) == MaturityLevel.DEVELOPING
    assert classify_maturity(3.5) == MaturityLevel.SCALING
    assert classify_maturity(4.3) == MaturityLevel.OPTIMIZING
    assert classify_maturity(5.0) == MaturityLevel.OPTIMIZING


def test_maturity_descriptions_for_all_levels():
    for level in MaturityLevel:
        desc = maturity_description(level)
        assert isinstance(desc, str) and len(desc) > 20
