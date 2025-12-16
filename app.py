import json
import os
import re
import google.generativeai as genai
from enum import Enum
from typing import List, Dict, Any

# AIOpsの判定ステータス
class HealthStatus(Enum):
    NORMAL = "GREEN"
    WARNING = "YELLOW"
    CRITICAL = "RED"

# app.pyからの呼び出し名に合わせてクラス名をLogicalRCAに変更
class LogicalRCA:
    def __init__(self, topology, config_dir: str = "./configs"):
        """
        LogicalRCA (旧 InferenceEngine) の初期化
        :param topology: トポロジー辞書オブジェクト または ファイルパス (str)
        :param config_dir: コンフィグファイルが格納されているディレクトリ (デフォルト: ./configs)
        """
        # topologyが文字列（ファイルパス）の場合はファイルから読み込み
        if isinstance(topology, str):
            self.topology = self._load_topology(topology)
        # 辞書の場合はそのまま使用
        elif isinstance(topology, dict):
            self.topology = topology
        else:
            raise ValueError("topology must be either a file path (str) or a dictionary")
        
        self.config_dir = config_dir
        
        # Google Generative AIの設定
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            # APIキーがない場合はエラーログを出力しつつ、動作を継続するか例外を投げる
            # ここでは明確に例外とする
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def _load_topology(self, path: str) -> Dict:
        """JSONファイルからトポロジー情報を読み込む"""
        if not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _read_config(self, device_id: str) -> str:
        """デバイスIDに対応するコンフィグファイルを読み込む"""
        config_path = os.path.join(self.config_dir, f"{device_id}.txt")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading config: {str(e)}"
        return "Config file not found."

    def _sanitize_text(self, text: str) -> str:
        """機密情報のサニタイズ処理"""
        # パスワード系 (Cisco/Junos)
        text = re.sub(r'(encrypted-password\s+)"[^"]+"', r'\1"********"', text)
        text = re.sub(r'(password|secret)\s+(\d)\s+\S+', r'\1 \2 ********', text)
        text = re.sub(r'(username\s+\S+\s+secret)\s+\d\s+\S+', r'\1 5 ********', text)
        # SNMP Community
        text = re.sub(r'(snmp-server community)\s+\S+', r'\1 ********', text)
        return text

    def analyze(self, alarms: List) -> List[Dict[str, Any]]:
        """
        アラームリストを分析して根本原因候補を返す
        app.pyから呼び出される主要なメソッド
        
        :param alarms: Alarmオブジェクトのリスト
        :return: 分析結果のリスト（辞書形式）
        """
        results = []
        
        if not alarms:
            return [{
                "id": "SYSTEM",
                "label": "No alerts detected",
                "prob": 0.0,
                "type": "Normal",
                "tier": 0
            }]
        
        # 各アラームに対して分析を実行
        for alarm in alarms:
            device_id = alarm.device_id
            alert_messages = [alarm.message]
            
            # LLMによる冗長性深度分析
            analysis = self.analyze_redundancy_depth(device_id, alert_messages)
            
            # 結果を整形
            prob = 0.5  # デフォルト
            if analysis["status"] == HealthStatus.CRITICAL:
                prob = 0.9
            elif analysis["status"] == HealthStatus.WARNING:
                prob = 0.7
            else:
                prob = 0.3
            
            results.append({
                "id": device_id,
                "label": alarm.message,
                "prob": prob,
                "type": analysis["impact_type"],
                "tier": 1 if prob > 0.6 else 2,
                "reason": analysis["reason"]
            })
        
        # 確率順にソート
        results.sort(key=lambda x: x["prob"], reverse=True)
        
        return results

    def analyze_redundancy_depth(self, device_id: str, alerts: List[str]) -> Dict[str, Any]:
        """
        LLMを使用して冗長性深度を判定する
        """
        if not alerts:
            return {
                "status": HealthStatus.NORMAL,
                "reason": "No active alerts detected.",
                "impact_type": "NONE"
            }

        device_info = self.topology.get(device_id, {})
        raw_config = self._read_config(device_id)
        
        # TOPOLOGYが辞書の場合、metadataの取得方法を調整
        if hasattr(device_info, 'metadata'):
            metadata = device_info.metadata
        elif isinstance(device_info, dict):
            metadata = device_info.get('metadata', {})
        else:
            metadata = {}

        # サニタイズ
        safe_config = self._sanitize_text(raw_config)
        safe_alerts = [self._sanitize_text(a) for a in alerts]

        prompt = f"""
あなたはネットワーク運用のエキスパートAIです。
以下の情報に基づき、現在発生しているアラートが「サービス停止(CRITICAL)」を引き起こしているか、
それとも「冗長機能によりサービスは維持されている(WARNING)」状態かを判定してください。

### 対象デバイス
- **Device ID**: {device_id}
- **Metadata**: {json.dumps(metadata)}

### 設定ファイル (Config - Sanitized)
```text
{safe_config}
```

### 発生中のアラートリスト
{json.dumps(safe_alerts)}

### 判定ルール (Thinking Process)
１．電源(PSU)障害の判定:
  ・デバイスが複数のPSUを持っていると推測され、かつ「全て」ではなく「一部」のPSUのみがFailしている場合。
  ・判定: WARNING (理由: Redundancy Lost - 片系運転中)
  ・全てのPSUがFail、または単一PSUデバイスのFailの場合。
  ・判定: CRITICAL (理由: Power Outage)
２．インターフェース/LAG障害の判定:
　・Configを確認し、Downしている物理インターフェースが LAG (Port-Channel / ae / Bond) のメンバーか確認してください。
　・親となる論理インターフェース(Port-Channel Xなど)自体のアラートが出ていなければ、親はUpしているとみなします。
　・メンバーのみのDownの場合。
　・判定: WARNING (理由: Degraded - 帯域縮退)
　・LAG非構成ポートのDown、または親LAG自体のDownの場合。
　・判定: CRITICAL (理由: Link Down - Service Impacting)
３．その他:
　・上記に当てはまらない不明なエラーや、CPU高負荷などは内容に応じて判断してください。

### 出力フォーマット
以下のJSON形式のみを出力してください。Markdownのコードブロック(```json ...)は含めないでください。
{{
  "status": "STATUS_STRING",
  "reason": "判定理由を簡潔に記述",
  "impact_type": "IMPACT_STRING"
}}

- status: "NORMAL", "WARNING", "CRITICAL" のいずれか
- impact_type: "NONE", "DEGRADED", "REDUNDANCY_LOST", "OUTAGE", "UNKNOWN" のいずれか
"""

        try:
            # LLMへの問い合わせ
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            # レスポンス解析
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            result_json = json.loads(response_text)
            
            status_str = result_json.get("status", "CRITICAL").upper()
            if status_str in ["GREEN", "NORMAL"]:
                health_status = HealthStatus.NORMAL
            elif status_str in ["YELLOW", "WARNING"]:
                health_status = HealthStatus.WARNING
            else:
                health_status = HealthStatus.CRITICAL

            return {
                "status": health_status,
                "reason": result_json.get("reason", "AI provided no reason"),
                "impact_type": result_json.get("impact_type", "UNKNOWN")
            }

        except Exception as e:
            print(f"[!] AI Inference Error: {e}")
            return {
                "status": HealthStatus.CRITICAL,
                "reason": f"AI Analysis Failed: {str(e)}",
                "impact_type": "AI_ERROR"
            }


if __name__ == "__main__":
    # テスト用設定
    TEST_TOPOLOGY = "topology.json"
    TEST_CONFIG_DIR = "./configs"

    # 簡易的なファイル生成（ディレクトリが存在しない場合のみ作成）
    if not os.path.exists(TEST_CONFIG_DIR):
        os.makedirs(TEST_CONFIG_DIR)

    try:
        # クラス名をLogicalRCAに変更して初期化
        engine = LogicalRCA(TEST_TOPOLOGY, TEST_CONFIG_DIR)
        
        print("--- AI Redundancy Analysis Test (LogicalRCA) ---")
        test_device = "WAN_ROUTER_01"
        test_alerts = ["Environment: PSU 1 Status Failed", "Environment: PSU 2 Status OK"]
        
        result = engine.analyze_redundancy_depth(test_device, test_alerts)
        print(f"Result: {result['status'].value} ({result['impact_type']})")
        print(f"Reason: {result['reason']}")

    except ValueError as e:
        print(f"Config Error: {e}")
    except Exception as e:
        print(f"Execution Error: {e}")
