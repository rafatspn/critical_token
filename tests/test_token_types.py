from ctlabel.select import token_type
from ctlabel.swap import top_alternatives


def test_latex_commands_are_matched_exactly():
    assert token_type(r"\left") == "latex_other"
    assert token_type(r"\le") == "operator"


def test_alternatives_include_argmax_when_original_is_not_argmax():
    assert top_alternatives([10, 11, 12, 13, 14], original_token_id=12) == [10, 11]
    assert top_alternatives([10, 11, 12, 13, 14], original_token_id=10) == [11, 12]
