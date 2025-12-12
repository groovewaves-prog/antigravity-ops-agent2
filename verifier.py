"""
Google Antigravity AIOps Agent - Verification Module (Universal)
特定のベンダーやコマンド形式に依存せず、普遍的なキーワードに基づいて
事実(Fact)を抽出する、メンテナンスフリーな検証モジュール。
"""
import re

def verify_log_content(log_text: str) -> dict:
    """
    ログテキストから、ベンダー非依存の共通キーワードを用いて客観的事実を抽出する
    """
    facts = {
        "ping_status": "Unknown",
        "interface_status": "Unknown",
        "hardware_status": "Unknown",
        "error_keywords": "None"
    }

    if not log_text:
        return facts

    # テキストを小文字化して検索（大文字小文字の揺れを吸収）
    text_lower = log_text.lower()

    # 1. Ping / 疎通確認 (普遍的な成功/失敗パターン)
    if "ping" in text_lower or "icmp" in text_lower:
        # 成功パターン: "100%", "!!!!!", "0% loss", "alive"
        if re.search(r'(100%|!!!!!|0% loss|alive)', text_lower):
            facts["ping_status"] = "OK (Success pattern found)"
        # 失敗パターン: "0%", ".....", "100% loss", "unreachable", "timed out"
        elif re.search(r'(0%|\.\.\.\.\.|100% loss|unreachable|timed out)', text_lower):
            facts["ping_status"] = "NG (Failure pattern found)"

    # 2. インターフェース / リンク状態
    # "down" という単語が "admin down" 以外で含まれるか
    if "down" in text_lower:
        if "administratively down" in text_lower:
             facts["interface_status"] = "Info (Admin Down detected)"
        else:
             facts["interface_status"] = "Critical (Link DOWN detected)"
    elif "up" in text_lower and "line protocol is up" in text_lower:
        facts["interface_status"] = "OK (UP detected)"

    # 3. ハードウェア / 環境 (Fan, Power, Temp)
    # "fail", "fault", "alarm" などの強い言葉を探す
    hw_keywords = r'(fan|power|temp|environment)'
    if re.search(hw_keywords, text_lower):
        if re.search(r'(fail|fault|error|critical|alert|problem)', text_lower):
            facts["hardware_status"] = "Critical (Hardware Failure pattern)"
        elif re.search(r'(ok|good|normal)', text_lower):
            facts["hardware_status"] = "OK"

    # 4. 汎用エラーワード検出 (予期せぬエラーのキャッチ)
    # これにより、定義していない未知の障害でも「なにかおかしい」ことを検知できる
    error_pattern = r'(error|fail|exception|denied|refused|panic|crash)'
    matches = re.findall(error_pattern, text_lower)
    if matches:
        # 重複排除してリスト化
        unique_errors = list(set(matches))
        facts["error_keywords"] = f"Detected: {', '.join(unique_errors)}"

    return facts

def format_verification_report(facts: dict) -> str:
    """
    検証結果をAIへの指示用テキストに変換する
    """
    return f"""
    【システム自動検証結果 (Ground Truth / Rule-Based)】
    ※以下の客観的事実と矛盾する回答をしてはならない。
    - 疎通判定 : {facts['ping_status']}
    - I/F状態  : {facts['interface_status']}
    - HW状態   : {facts['hardware_status']}
    - 危険単語 : {facts['error_keywords']}
    """
