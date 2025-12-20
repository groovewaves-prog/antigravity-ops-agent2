"""
Google Antigravity AIOps Agent - Network Operations Module (v2 - 原因分析 & Remediation分離版)
"""
import re
import os
import time
import json
import google.generativeai as genai
from netmiko import ConnectHandler

SANDBOX_DEVICE = {
    'device_type': 'cisco_nxos',
    'host': 'sandbox-nxos-1.cisco.com',
    'username': 'admin',
    'password': 'Admin_1234!',
    'port': 22,
    'global_delay_factor': 2,
    'banner_timeout': 30,
    'conn_timeout': 30,
}

def sanitize_output(text: str) -> str:
    rules = [
        (r'(password|secret) \d+ \S+', r'\1 <HIDDEN_PASSWORD>'),
        (r'(encrypted password) \S+', r'\1 <HIDDEN_PASSWORD>'),
        (r'(snmp-server community) \S+', r'\1 <HIDDEN_COMMUNITY>'),
        (r'(username \S+ privilege \d+ secret \d+) \S+', r'\1 <HIDDEN_SECRET>'),
        (r'\b(?!(?:10|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.)\d{1,3}\.(?:\d{1,3}\.){2}\d{1,3}\b', '<MASKED_PUBLIC_IP>'),
        (r'([0-9A-Fa-f]{4}\.){2}[0-9A-Fa-f]{4}', '<MASKED_MAC>'),
    ]
    for pattern, replacement in rules:
        text = re.sub(pattern, replacement, text)
    return text

# =====================================================
# ★新規: 原因分析専用プロンプト
# =====================================================
def generate_analyst_report(scenario, target_node, topology_context, target_conf, verification_context, api_key):
    """
    【原因分析専用】
    障害の「なぜ起きたか」を分析するレポート。
    復旧手順・確認コマンドは含めない（Remediation側で処理）。
    
    Args:
        scenario: 障害シナリオ名
        target_node: 対象デバイスのNodeオブジェクト
        topology_context: トポロジーコンテキスト（親・子・CI情報）
        target_conf: 設定ファイル内容
        verification_context: 検証ログ
        api_key: API Key
    
    Returns:
        str: Markdown形式の原因分析レポート
    """
    if not api_key:
        return "Error: API Key Missing"
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemma-3-12b-it", generation_config={"temperature": 0.0})
    
    prompt = f"""
あなたはネットワーク障害分析専門のAIアナリストです。
以下の事実のみから「原因分析レポート」を作成してください。

【禁止事項】
- 復旧手順やコマンドを書かない
- 対応方法を書かない
- 「現在、原因究明と復旧作業を最優先で進めております」などの定型句を書かない
- 「今後の対応」セクションは不要

【必須構成】以下の見出しを必ず含めてください（見出し文言は変更しない）:
1. 障害概要
2. 発生原因（最重要：なぜこの障害が起きたのか、技術的背景）
3. 影響範囲
4. 技術的根拠
5. 切り分け判断の理由

【出力形式】
- Markdown形式
- 文体は「です/ます調」で統一
- 不明な点は「未確認」、推測は「推定」と明示
- コードブロック（```）は使用しない

【入力情報】
- シナリオ: {scenario}
- 対象機器: {target_node.id} ({target_node.metadata.get('vendor', '不明')} {target_node.metadata.get('os', '不明')} {target_node.metadata.get('model', '')})
- CI/トポロジー: {json.dumps(topology_context, ensure_ascii=False)}
- Config(抜粋): {(target_conf or 'なし')[:2000]}
- 検証ログ: {verification_context}
"""

    try:
        response = model.generate_content(prompt)
        return response.text if hasattr(response, "text") and response.text else str(response)
    except Exception as e:
        return f"分析レポート生成エラー: {type(e).__name__}: {e}"


# =====================================================
# ★改修: Remediation専用プロンプト（復旧手順・確認コマンドのみ）
# =====================================================
def generate_remediation_commands(scenario, analysis_result, target_node, api_key):
    """
    【復旧手順専用】
    「どう直すか」「どう確認するか」に特化。
    原因分析の説明は不要（AI Analyst Report側で完結）。
    
    Args:
        scenario: 障害シナリオ名
        analysis_result: AI Analyst Report からの要約・結論
        target_node: 対象デバイスのNodeオブジェクト
        api_key: API Key
    
    Returns:
        str: Markdown形式の復旧手順
    """
    if not api_key: 
        return "Error: API Key Missing"
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemma-3-12b-it", generation_config={"temperature": 0.0})
    
    prompt = f"""
あなたは熟練したネットワーク運用者/エンジニアです。
以下の情報を使って、オペレーター向けの **復旧手順** を作成してください。

【重要】
- 原因分析の説明は不要です（AI Analyst Report側で既に完結）
- 前提条件として「【前提】原因は{scenario}と推定されます」の1行だけ述べること
- 後は「どうやって直すか」に専念してください

【入力】
- 対象デバイス: {target_node.id} ({target_node.metadata.get('vendor')} {target_node.metadata.get('os')} {target_node.metadata.get('model', '')})
- 発生シナリオ: {scenario}
- AI Analyst Report からの分析結果:
{analysis_result[:1500]}

【出力】（です/ます調、運用者向け）
以下の見出しで、**実施手順と確認のみ** を出力してください：

### 復旧手順（Remediation）

**【前提】** 原因は{scenario}と推定されます。

#### 1. 実施前提・注意点
停止判断などのSafety-Criticalな項目は、運用者（人間）が最終判断する前提で記述してください。
- 実施前の確認事項
- リスク評価
- ロールバック可能性

#### 2. 設定バックアップ手順
バックアップを取得する方法を複数提示します。
```bash
（バックアップコマンド例）
```

#### 3. 復旧手順
段階的に実行する手順を記述してください。各段階でコマンドがあれば以下の形式で：
```bash
（復旧コマンド例）
```

#### 4. ロールバック手順
失敗時の戻し方法を記述してください。

#### 5. 正常性確認コマンド
復旧後、正常に戻ったかを確認するコマンドを記述してください。
```bash
（確認コマンド例）
```

期待結果（合否判定のキーワード）も明示してください。

※ 情報不足な場合は、推定できる範囲の最小セットを提示し、
   「追加で必要な情報」として最後に列挙してください。
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text if hasattr(response, "text") and response.text else str(response)
    except Exception as e:
        return f"Remediation生成エラー: {type(e).__name__}: {e}"


# =====================================================
# 既存関数（変更なし）
# =====================================================

def generate_fake_log_by_ai(scenario_name, target_node, api_key):
    """
    シナリオ名と機器メタデータから、AIが自律的に障害ログを生成する
    """
    if not api_key: return "Error: API Key Missing"
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        "gemma-3-12b-it",
        generation_config={"temperature": 0.2}
    )
    
    vendor = target_node.metadata.get("vendor", "Generic")
    os_type = target_node.metadata.get("os", "Generic OS")
    model_name = target_node.metadata.get("model", "Generic Device")
    hostname = target_node.id

    prompt = f"""
    あなたはネットワーク機器のCLIシミュレーター（熟練エンジニアのロールプレイング）です。
    ユーザーが指定した「障害シナリオ」に基づいて、トラブルシューティング時に実行されるであろう
    **「コマンド」とその「実行結果ログ」** を生成してください。

    【入力情報】
    - 対象ホスト名: {hostname}
    - ベンダー: {vendor}
    - OS種別: {os_type}
    - モデル: {model_name}
    - **発生している障害シナリオ**: 「{scenario_name}」

    【AIへの指示】
    1. **シナリオの解釈**: 提供されたシナリオ名から、技術的にどのような状態であるべきか推測してください。
    2. **コマンド選択**: その障害を確認するために、このベンダー({vendor})でよく使われる確認コマンドを2〜3個選んでください。
    3. **ログ生成**: 選んだコマンドに対し、シナリオ通りの異常状態を示す出力を生成してください。
    4. **リアリティ**: タイムスタンプやプロンプトを含め、本物のCLI画面のように出力してください。

    【出力形式】
    解説不要。CLIのテキストデータのみを出力してください。
    Markdownのコードブロックは使用しないでください。
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Generation Error: {e}"

def generate_config_from_intent(target_node, current_config, intent_text, api_key):
    if not api_key: return "Error: API Key Missing"
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemma-3-12b-it", generation_config={"temperature": 0.0})
    
    vendor = target_node.metadata.get("vendor", "Unknown Vendor")
    os_type = target_node.metadata.get("os", "Unknown OS")
    
    prompt = f"""
    ネットワーク設定生成。
    対象: {target_node.id} ({vendor} {os_type})
    現在のConfig: {current_config}
    Intent: {intent_text}
    出力: 投入用コマンドのみ (Markdownコードブロック)
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Config Gen Error: {e}"

def generate_health_check_commands(target_node, api_key):
    if not api_key: return "Error: API Key Missing"
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemma-3-12b-it", generation_config={"temperature": 0.0})
    
    vendor = target_node.metadata.get("vendor", "Unknown Vendor")
    os_type = target_node.metadata.get("os", "Unknown OS")
    
    prompt = f"Netmiko正常性確認コマンドを3つ生成せよ。対象: {vendor} {os_type}。出力: コマンドのみ箇条書き"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Command Gen Error: {e}"

def run_diagnostic_simulation(scenario_type, target_node=None, api_key=None):
    time.sleep(1.5)
    
    if "---" in scenario_type or "正常" in scenario_type:
        return {"status": "SKIPPED", "sanitized_log": "No action required.", "error": None}

    if "[Live]" in scenario_type:
        commands = ["terminal length 0", "show version", "show interface brief", "show ip route"]
        try:
            with ConnectHandler(**SANDBOX_DEVICE) as ssh:
                if not ssh.check_enable_mode(): ssh.enable()
                prompt = ssh.find_prompt()
                raw_output = f"Connected to: {prompt}\n"
                for cmd in commands:
                    output = ssh.send_command(cmd)
                    raw_output += f"\n{'='*30}\n[Command] {cmd}\n{output}\n"
        except Exception as e:
            return {"status": "ERROR", "sanitized_log": "", "error": str(e)}
        return {"status": "SUCCESS", "sanitized_log": sanitize_output(raw_output), "error": None}
            
    elif "全回線断" in scenario_type or "サイレント" in scenario_type or "両系" in scenario_type:
        return {"status": "ERROR", "sanitized_log": "", "error": "Connection timed out"}

    else:
        if api_key and target_node:
            raw_output = generate_fake_log_by_ai(scenario_type, target_node, api_key)
            return {"status": "SUCCESS", "sanitized_log": sanitize_output(raw_output), "error": None}
        else:
            return {"status": "ERROR", "sanitized_log": "", "error": "API Key or Target Node Missing"}

def predict_initial_symptoms(scenario_name, api_key):
    """
    障害シナリオ名から、発生しうる「初期症状」を推論
    """
    if not api_key: return {}
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemma-3-12b-it", generation_config={"temperature": 0.0})
    
    prompt = f"""
    あなたはネットワーク監視システムのAIエージェントです。
    指定された「障害シナリオ」において、監視システムが最初に検知するであろう「初期症状」を推論してください。

    **シナリオ**: {scenario_name}

    【出力要件】
    1. 以下のキーを持つ **JSON形式** で出力すること。解説は不要。
       - "alarm": アラームメッセージ
       - "ping": 疎通状態
       - "log": ログキーワード
    
    2. 値は適切なものを選んでください。

    **例**:
    シナリオ: "[WAN] BGPルートフラッピング"
    出力: {{ "alarm": "BGP Flapping", "ping": "OK", "log": "" }}
    """
    
    try:
        response = model.generate_content(prompt)
        text = response.text
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"Symptom Prediction Error: {e}")
        return {}
