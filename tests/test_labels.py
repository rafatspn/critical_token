from ctlabel.labels import finalize_position_labels, label_position


def label(**overrides):
    kwargs = dict(
        correct_trace=False, position=3, crit_cdpo=None, crit_good_candidate=None,
        with_count=(5, 100), before_count=(80, 100), sampled_count=None,
        alt2_count=None, alt3_count=None, tau_crit=0.3, zero_frac=3 / 64,
    )
    kwargs.update(overrides)
    return label_position(**kwargs)


def test_critical_drop_and_cdpo():
    result = label(crit_cdpo=3)
    assert result["is_crit_cdpo"] and result["is_crit_drop"]


def test_fixable():
    assert label(with_count=(0, 64), before_count=(0, 64), sampled_count=(32, 64))["is_fixable"]


def test_consequence():
    assert label(position=5, crit_cdpo=3, with_count=(2, 64), before_count=(1, 64))["is_consequence"]


def test_good_critical():
    result = label(correct_trace=True, with_count=(90, 100), before_count=(40, 100), alt2_count=(10, 100), alt3_count=(20, 100))
    assert result["is_crit_good"]


def test_noncritical_and_ambiguous():
    neutral = label(with_count=(4000, 10000), before_count=(4000, 10000))
    neutral["zero_frac"] = 3 / 64
    assert finalize_position_labels(neutral, before_critical=True, tau_null=0.1)["is_noncritical"]
    unknown = label(with_count=None, before_count=None)
    unknown["zero_frac"] = 3 / 64
    assert finalize_position_labels(unknown, before_critical=True, tau_null=0.1)["is_ambiguous"]

