# tests/test_process_sim.py

"""
変更点と解説:

mock_cover_get_next の修正: 
side_effect で置き換えるモック関数 mock_cover_get_next は、元の cover_get_next メソッドを 
data_dict_like_id_value=True で呼び出す役割に留めます。
Process インスタンスのモック: 
Process クラスのインスタンス自体をモック化し、multi 関数がそのモックインスタンスを使用するようにします。
返り値の型の動的検証:
mock_process_instance.cover_get_next.call_args_list をループ処理し、各 cover_get_next の呼び出し時
の引数 (called_with_data_dict) を取得します。
original_cover_get_next を使用して、モックインスタンスと取得した引数、そして強制的に data_dict_like_id_value=True 
を渡して元のメソッドを呼び出し、実際の返り値を取得します。
取得した返り値の辞書に対して、キーが int 型、値が float 型であることを検証します。
空の辞書の場合は型チェックをスキップしています。
最終結果の型と高レベルな振る舞いの検証: 
multi 関数の最終的な結果の型 (float) をアサートします。
具体的な値は動的な cover_get_next の振る舞いに依存するため、ここでは型のみを保証するか、もしあ
れば結果に対するより高レベルな振る舞い（例：非負であることなど）をアサートします。

この修正により、テストの実行時のみ cover_get_next メソッドが data_dict_like_id_value=True で動作す
るように振る舞いを変更し、その動的な返り値である辞書のキーと値の型を各呼び出しごとに検証することがで
きます。最終的な multi 関数の具体的な値は、内部の処理に依存するため、必要に応じてより高レベルな振る
舞いをアサートに追加してください。
"""
import pytest
from process_sim import multi, Process
from unittest import mock

def test_multi_with_forced_cover_get_next_id_value_true_and_dynamic_return_type_check(mocker):
    """
    multi 関数を実行し、テスト時のみ cover_get_next が
    data_dict_like_id_value=True で振る舞うようにモックし、
    その返り値の辞書のキーと値の型を検証する（値は動的）。
    """
    original_cover_get_next = Process.cover_get_next

    def mock_cover_get_next(self, data_dict, data_dict_like_id_value=False):
        # テスト時のみ data_dict_like_id_value を True に強制して元のメソッドを呼び出す
        return original_cover_get_next(self, data_dict, data_dict_like_id_value=True)

    # Process クラスの cover_get_next メソッドをモック関数で置き換える
    mocker.patch.object(Process, 'cover_get_next', side_effect=mock_cover_get_next)

    # Process クラスのインスタンスをモック化する
    mock_process_instance = mocker.MagicMock(spec=Process)

    # multi 関数が Process のインスタンスを作成する際に、このモックを返すようにする
    mocker.patch('process_sim.Process', return_value=mock_process_instance)

    result = multi()

    # cover_get_next が multi 関数内のループ回数 * sim 関数内のループ回数 だけ呼ばれることをアサート
    assert mock_process_instance.cover_get_next.call_count == 4 * 3  # multi の m=4, sim の n=3

    # 各 cover_get_next の呼び出し後の返り値の型を検証
    for call in mock_process_instance.cover_get_next.call_args_list:
        args, kwargs = call
        # cover_get_next は self と data_dict を引数に取る
        called_with_data_dict = args[1]
        returned_dict = original_cover_get_next(mock_process_instance, called_with_data_dict, data_dict_like_id_value=True)

        if returned_dict:  # 空の辞書の場合の型チェックはスキップ
            assert all(isinstance(key, int) for key in returned_dict.keys())
            assert all(isinstance(value, float) for value in returned_dict.values())

    # multi 関数の最終的な結果の型をアサート (float)
    assert isinstance(result, float)

    # 正確な最終結果の値は cover_get_next の動的な振る舞いに依存するため、
    # ここでは型のみを保証する、またはより高レベルな振る舞いをアサートする
    # (例: 値が非負であるなど、もしあれば)
    assert result >= 0  # 例：結果が非負であることを確認 (具体的な振る舞いに応じて変更)
