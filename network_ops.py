"""
Google Antigravity AIOps Agent - Network Operations Module
"""
import re
import os
import time
import google.generativeai as genai
from netmiko import ConnectHandler

# Cisco DevNet Sandbox (Nexus 9000)
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
    """AIにそれっぽいログを捏造させる"""
    if not api_key: return "Error: API Key Missing"
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    prompt = f"""
    Cisco IOS/NX-OS のコマンド実行結果（ログ）を生成してください。
    シナリオ: {scenario_name}
    対象: WAN_ROUTER_01
    要件: 生ログのみ出力。解説不要。障害を示すエラーログを含めること。
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Generation Error: {e}"

def run_diagnostic_simulation(scenario_type, api_key=None):
    """
    診断実行関数 (修正版: api_key引数を明示)
    """
    time.sleep(1.5)
    
    status = "SUCCESS"
    raw_output = ""
    error_msg = None

    # 区切り線や無効な選択の場合
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
