"""Verify the skeleton package can be imported."""


def test_package_import():
    import poker_decision

    assert poker_decision.__name__ == "poker_decision"
