"""Config join disable for fair n:m vs 1:1 comparison."""

from phonebox.core.vectorizer import Vectorizer


def test_disable_config_joins_clears_regexes():
    vec = Vectorizer(locale="it_IT", phoneset_name="ipa")
    assert vec.lett_join_re is not None
    assert vec.phon_join_re is not None
    vec.disable_config_joins()
    assert vec.lett_join_re is None
    assert vec.phon_join_re is None
    cooked = vec.cook_letters("chi", g2p=True)
    assert "₊" not in "".join(cooked)
    assert cooked == ["c", "h", "i"]


def test_pt_joined_when_enabled():
    vec = Vectorizer(locale="pt_BR", phoneset_name="ipa")
    cooked = vec.cook_letters("que", g2p=True)
    assert any("₊" in t for t in cooked)
