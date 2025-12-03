"""
Google Antigravity AIOps Agent - Network Operations Module
"""
import re
import os
import time
import google.generativeai as genai
from netmiko import ConnectHandler

# Cisco DevNet Sandbox
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

def generate_fake_log_by_ai(scenario_name, api_key):
    """
    シナリオに応じた適切な制約条件を与えてログを生成する
    """
    if not api_key: return "Error: API Key Missing"
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # シナリオごとの厳格な制約条件 (AIの暴走防止)
    constraints = ""
    
    if "片系" in scenario_name or "FAN" in scenario_name or "メモリ" in scenario_name:
        constraints = """
        【重要：ログ生成の絶対条件】
        1. **全てのインターフェース(GigabitEthernet等)は UP/UP (正常) にすること。**
        2. Ping疎通は **100% 成功 (!!!!!)** すること。
        3. ルーティングテーブルは正常に表示すること。
        4. 障害は `show environment` や `show processes memory` の結果、および `syslog` メッセージのみで表現すること。
        5. 通信断を示唆するログは一切含めないこと。
        """
    elif "全回線断" in scenario_name or "両系" in scenario_name:
        constraints = """
        【重要：ログ生成の絶対条件】
        1. 主要インターフェースは **DOWN/DOWN** または **Administratively Down** にすること。
        2. Ping疎通は **0% (.....)** にすること。
        """
    elif "BGP" in scenario_name:
        constraints = """
        【重要：ログ生成の絶対条件】
        1. 物理インターフェースは **UP/UP** にすること。
        2. Pingは通ること。
        3. `show ip bgp summary` の State が Idle または Active (確立中) でフラついている様子を見せること。
        """

    prompt = f"""
    あなたはCiscoネットワーク機器のシミュレーターです。
    以下のシナリオに基づき、エンジニアが調査した際のコマンド実行ログを生成してください。

    **発生シナリオ**: {scenario_name}
    **対象機器**: Cisco IOS Router (WAN_ROUTER_01)

    {constraints}

    **出力すべきコマンド例**:
    - `show version`
    - `show ip interface brief`
    - `show environment` (ハードウェア障害の場合)
    - `show log` (エラーメッセージ)
    - `ping <対向IP>`

    **出力形式**:
    解説やMarkdownの装飾は不要です。CLIの生テキストのみを出力してください。
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Generation Error: {e}"

def run_diagnostic_simulation(scenario_type, api_key=None):
    """診断実行関数"""
    time.sleep(1.5)
    
    status = "SUCCESS"
    raw_output = ""
    error_msg = None

    if "---" in scenario_type or "正常" in scenario_type:
        return {"status": "SKIPPED", "sanitized_log": "No action required.", "error": None}

    # Live実機診断
    if "[Live]" in scenario_type:
        commands = ["terminal length 0", "show version", "show interface brief", "show ip route"]
        try:
            with ConnectHandler(**SANDBOX_DEVICE) as ssh:
                if not ssh.check_enable_mode(): ssh.enable()
                prompt = ssh.find_prompt()
                raw_output += f"Connected to: {prompt}\n"
                for cmd in commands:
                    output = ssh.send_command(cmd)
                    raw_output += f"\n{'='*30}\n[Command] {cmd}\n{output}\n"
        except Exception as e:
            status = "ERROR"
            error_msg = str(e)
            raw_output = f"Real Device Connection Failed: {error_msg}"
            
    # 全断・サイレント（接続不可系）
    elif "全回線断" in scenario_type or "サイレント" in scenario_type:
        status = "ERROR"
        error_msg = "Connection timed out"
        raw_output = "SSH Connection Failed. Host Unreachable."

    # その他（AI生成）
    else:
        if api_key:
            raw_output = generate_fake_log_by_ai(scenario_type, api_key)
        else:
            status = "ERROR"
            error_msg = "API Key Required"
            raw_output = "Cannot generate logs without API Key."

    return {
        "status": status,
        "sanitized_log": sanitize_output(raw_output),
        "error": error_msg
    }
