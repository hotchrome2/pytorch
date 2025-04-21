import pytest
from your_module import Simu, Inter, multi_sim

def test_cover_process_and_inter_behavior(monkeypatch):
    cover_count = 0
    trans_count = 0
    inverse_count = 0

    # Simu.cover_process をラップして、呼び出し回数と戻り値の型チェック
    original_cover = Simu.cover_process
    def wrapped_cover(self, in1, in2):
        nonlocal cover_count
        cover_count += 1
        result = original_cover(self, in1, in2)

        # 戻り値が辞書であることを確認
        assert isinstance(result, dict), "cover_process の戻り値は dict であるべきです"

        for k, v in result.items():
            assert isinstance(k, int), f"辞書のキーが int でない: {k} ({type(k)})"
            assert isinstance(v, float), f"辞書の値が float でない: {v} ({type(v)})"

        return result

    # Inter.__init__ をラップして trans / inverse を記録するように置換
    original_init = Inter.__init__
    original_trans = Inter.trans
    original_inverse = Inter.inverse

    def wrapped_init(self):
        original_init(self)

        def wrapped_trans(x):
            nonlocal trans_count
            trans_count += 1
            return original_trans(self, x)

        def wrapped_inverse(x):
            nonlocal inverse_count
            inverse_count += 1
            return original_inverse(self, x)

        self.trans = wrapped_trans
        self.inverse = wrapped_inverse

    # monkeypatch の適用
    monkeypatch.setattr(Simu, "cover_process", wrapped_cover)
    monkeypatch.setattr(Inter, "__init__", wrapped_init)

    # multi_sim の実行（内部で cover_process, trans, inverse を呼び出す）
    _ = multi_sim()

    # 検証：呼び出し回数チェック
    assert cover_count > 0, "cover_process が一度も呼ばれていません"
    assert trans_count == cover_count, f"trans の呼び出し回数({trans_count}) ≠ cover_process({cover_count})"
    assert inverse_count == cover_count, f"inverse の呼び出し回数({inverse_count}) ≠ cover_process({cover_count})"
