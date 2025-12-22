# -*- coding: utf-8 -*-
"""
AIOps Agent - Inference Engine (Improved v6)
=============================================
改善点:
1. gemma-3-12b-it モデルに統一
2. バッチ処理対応（複数デバイスを1リクエストで処理）
3. グローバルレートリミッター統合
4. キャッシュ活用の強化
5. ローカルルール優先（LLM呼び出し削減）
"""

import json
import os
import re
import logging
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import google.generativeai as genai

# レートリミッターのインポート
from rate_limiter import (
    GlobalRateLimiter,
    rate_limited_with_retry,
    estimate_tokens,
    check_input_limit
)

logger = logging.getLogger(__name__)

# =====================================================
# 定数・設定
# =====================================================
MODEL_NAME = "gemma-3-12b-it"  # ★統一されたモデル名
MAX_BATCH_SIZE = 5             # バッチ処理の最大デバイス数
MAX_PROMPT_TOKENS = 100000     # プロンプト最大トークン数（安全マージン込み）


class HealthStatus(Enum):
    NORMAL = "GREEN"
    WARNING = "YELLOW"
    CRITICAL = "RED"


@dataclass
class AnalysisResult:
    """分析結果を格納するデータクラス"""
    device_id: str
    status: HealthStatus
    reason: str
    impact_type: str
    confidence: float = 0.0
    from_cache: bool = False
    from_local_rule: bool = False


# =====================================================
# LogicalRCA クラス (改善版)
# =====================================================
class LogicalRCA:
    """
    LogicalRCA (v6 - Improved):
      - バッチ処理対応
      - レートリミッター統合
      - キャッシュ活用強化
      - ローカルルール優先
    """

    # サイレント障害推定の閾値
    SILENT_MIN_CHILDREN = 2
    SILENT_RATIO = 0.5

    def __init__(self, topology, config_dir: str = "./configs"):
        if isinstance(topology, str):
            self.topology = self._load_topology(topology)
        elif isinstance(topology, dict):
            self.topology = topology
        else:
            raise ValueError("topology must be either a file path (str) or a dictionary")

        self.config_dir = config_dir
        self.model = None
        self._api_configured = False
        self._rate_limiter = GlobalRateLimiter()

        # parent -> [children...] マップを構築
        self.children_map: Dict[str, List[str]] = {}
        for dev_id, info in self.topology.items():
            p = self._get_parent_id_from_info(info)
            if p:
                self.children_map.setdefault(p, []).append(dev_id)
        
        logger.info(f"LogicalRCA initialized with {len(self.topology)} nodes")

    # ----------------------------
    # Topology helpers
    # ----------------------------
    def _get_parent_id_from_info(self, info: Any) -> Optional[str]:
        """ノード情報から親IDを取得"""
        if isinstance(info, dict):
            return info.get("parent_id")
        if hasattr(info, "parent_id"):
            return getattr(info, "parent_id")
        return None

    def _get_device_info(self, device_id: str) -> Any:
        return self.topology.get(device_id, {})

    def _get_parent_id(self, device_id: str) -> Optional[str]:
        info = self._get_device_info(device_id)
        return self._get_parent_id_from_info(info)

    def _get_metadata(self, device_id: str) -> Dict[str, Any]:
        info = self._get_device_info(device_id)
        if isinstance(info, dict):
            md = info.get("metadata", {})
            return md if isinstance(md, dict) else {}
        if hasattr(info, "metadata"):
            md = getattr(info, "metadata")
            return md if isinstance(md, dict) else {}
        if hasattr(info, "get_metadata"):
            try:
                return info.get_metadata("metadata", {})
            except Exception:
                return {}
        return {}

    def _get_psu_count(self, device_id: str, default: int = 1) -> int:
        md = self._get_metadata(device_id)
        if isinstance(md, dict):
            hw = md.get("hw_inventory", {})
            if isinstance(hw, dict) and "psu_count" in hw:
                try:
                    return int(hw.get("psu_count"))
                except Exception:
                    pass
            if str(md.get("redundancy_type", "")).upper() == "PSU":
                return 2
        return default

    # ----------------------------
    # LLM init
    # ----------------------------
    def _ensure_api_configured(self) -> bool:
        if self._api_configured:
            return True
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return False
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(MODEL_NAME)
            self._api_configured = True
            logger.info(f"API configured with model: {MODEL_NAME}")
            return True
        except Exception as e:
            logger.error(f"API Configuration Error: {e}")
            return False

    # ----------------------------
    # IO
    # ----------------------------
    def _load_topology(self, path: str) -> Dict:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _read_config(self, device_id: str) -> str:
        config_path = os.path.join(self.config_dir, f"{device_id}.txt")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return f.read()[:2000]  # 最大2000文字に制限
            except Exception as e:
                return f"Error reading config: {str(e)}"
        return "Config file not found."

    # ----------------------------
    # Sanitization
    # ----------------------------
    def _sanitize_text(self, text: str) -> str:
        text = re.sub(r'(encrypted-password\s+)"[^"]+"', r'\1"********"', text)
        text = re.sub(r"(password|secret)\s+(\d)\s+\S+", r"\1 \2 ********", text)
        text = re.sub(r"(username\s+\S+\s+secret)\s+\d\s+\S+", r"\1 5 ********", text)
        text = re.sub(r"(snmp-server community)\s+\S+", r"\1 ********", text)
        return text

    # ==========================================================
    # ★★★ ローカルルール（LLM呼び出し削減の核心） ★★★
    # ==========================================================
    def _apply_local_rules(self, device_id: str, alerts: List[str]) -> Optional[AnalysisResult]:
        """
        ローカルルールで判定可能なケースを処理（LLM呼び出しを回避）
        
        Returns:
            AnalysisResult if determined locally, None if LLM needed
        """
        if not alerts:
            return AnalysisResult(
                device_id=device_id,
                status=HealthStatus.NORMAL,
                reason="No active alerts detected.",
                impact_type="NONE",
                confidence=1.0,
                from_local_rule=True
            )

        safe_alerts = [self._sanitize_text(a) for a in alerts]
        joined = " ".join(safe_alerts)
        joined_lower = joined.lower()

        # ルール0: 停止系（赤）- 確実にCRITICAL
        critical_patterns = [
            ("Power Supply: Dual Loss", "Device down / dual PSU loss detected"),
            ("Dual Loss", "Dual power loss detected"),
            ("Device Down", "Device is completely down"),
            ("Thermal Shutdown", "Thermal shutdown - device offline"),
        ]
        for pattern, reason in critical_patterns:
            if pattern in joined or pattern.lower() in joined_lower:
                return AnalysisResult(
                    device_id=device_id,
                    status=HealthStatus.CRITICAL,
                    reason=f"{reason} (local safety rule).",
                    impact_type="Hardware/Physical",
                    confidence=0.95,
                    from_local_rule=True
                )

        # ルール1: 電源片系（黄色/赤）
        psu_count = self._get_psu_count(device_id, default=1)
        psu_single_fail = (
            ("power supply" in joined_lower and "failed" in joined_lower and "dual" not in joined_lower) 
            or ("psu" in joined_lower and "fail" in joined_lower and "dual" not in joined_lower)
        )
        if psu_single_fail:
            if psu_count >= 2:
                return AnalysisResult(
                    device_id=device_id,
                    status=HealthStatus.WARNING,
                    reason=f"Single PSU failure with redundancy (psu_count={psu_count}) (local safety rule).",
                    impact_type="Hardware/Redundancy",
                    confidence=0.9,
                    from_local_rule=True
                )
            return AnalysisResult(
                device_id=device_id,
                status=HealthStatus.CRITICAL,
                reason=f"Single PSU failure without redundancy (psu_count={psu_count}) (local safety rule).",
                impact_type="Hardware/Physical",
                confidence=0.9,
                from_local_rule=True
            )

        # ルール2: FAN（黄色 / 熱兆候で赤）
        fan_fail = ("fan fail" in joined_lower) or ("fan" in joined_lower and "fail" in joined_lower)
        overheat_hint = ("high temperature" in joined_lower) or ("overheat" in joined_lower) or ("thermal" in joined_lower)
        if fan_fail:
            if overheat_hint:
                return AnalysisResult(
                    device_id=device_id,
                    status=HealthStatus.CRITICAL,
                    reason="Fan failure with overheat/thermal symptom detected (local safety rule).",
                    impact_type="Hardware/Physical",
                    confidence=0.9,
                    from_local_rule=True
                )
            return AnalysisResult(
                device_id=device_id,
                status=HealthStatus.WARNING,
                reason="Fan failure detected. Service likely continues but risk of thermal escalation (local safety rule).",
                impact_type="Hardware/Degraded",
                confidence=0.85,
                from_local_rule=True
            )

        # ルール3: メモリ（黄色 / OOMで赤）
        mem_symptom = ("memory high" in joined_lower) or ("memory leak" in joined_lower)
        oom_hint = ("out of memory" in joined_lower) or ("oom" in joined_lower) or ("killed process" in joined_lower)
        if mem_symptom:
            if oom_hint:
                return AnalysisResult(
                    device_id=device_id,
                    status=HealthStatus.CRITICAL,
                    reason="Memory leak/high with OOM/crash symptom detected (local safety rule).",
                    impact_type="Software/Resource",
                    confidence=0.9,
                    from_local_rule=True
                )
            return AnalysisResult(
                device_id=device_id,
                status=HealthStatus.WARNING,
                reason="Memory high/leak symptom detected. Likely degraded but not down yet (local safety rule).",
                impact_type="Software/Resource",
                confidence=0.8,
                from_local_rule=True
            )

        # ルール4: インターフェースダウン
        if "interface down" in joined_lower or "link down" in joined_lower:
            return AnalysisResult(
                device_id=device_id,
                status=HealthStatus.WARNING,
                reason="Interface/Link down detected (local rule).",
                impact_type="Network/LinkDown",
                confidence=0.85,
                from_local_rule=True
            )

        # ルール5: BGPフラッピング
        if "bgp flapping" in joined_lower or "bgp peer down" in joined_lower:
            return AnalysisResult(
                device_id=device_id,
                status=HealthStatus.WARNING,
                reason="BGP instability detected (local rule).",
                impact_type="Network/BGP",
                confidence=0.85,
                from_local_rule=True
            )

        # ローカルルールで判定できない場合はNone
        return None

    # ==========================================================
    # ★★★ バッチ処理対応のLLM呼び出し ★★★
    # ==========================================================
    def _analyze_batch_with_llm(
        self, 
        devices_alerts: Dict[str, List[str]]
    ) -> Dict[str, AnalysisResult]:
        """
        複数デバイスを1つのLLMリクエストで分析
        
        Args:
            devices_alerts: {device_id: [alerts...], ...}
        
        Returns:
            {device_id: AnalysisResult, ...}
        """
        if not self._ensure_api_configured():
            # API未設定時のフォールバック
            return {
                dev_id: AnalysisResult(
                    device_id=dev_id,
                    status=HealthStatus.WARNING,
                    reason="API key not configured. Manual analysis required.",
                    impact_type="UNKNOWN",
                    confidence=0.3
                )
                for dev_id in devices_alerts
            }

        # キャッシュチェック
        cache_key = self._rate_limiter.compute_cache_key(
            "batch_analysis", 
            json.dumps(devices_alerts, sort_keys=True)
        )
        cached = self._rate_limiter.get_cache(cache_key)
        if cached:
            logger.info(f"Batch analysis cache hit for {len(devices_alerts)} devices")
            return cached

        # バッチプロンプト構築
        devices_info = []
        for dev_id, alerts in devices_alerts.items():
            metadata = self._get_metadata(dev_id)
            safe_alerts = [self._sanitize_text(a) for a in alerts]
            devices_info.append({
                "device_id": dev_id,
                "metadata": metadata,
                "alerts": safe_alerts
            })

        prompt = f"""
あなたはネットワーク運用のエキスパートAIです。
以下の複数デバイスについて、発生中のアラートがサービス停止(CRITICAL)を引き起こしているか、
冗長機能によりサービス維持(WARNING)されているか、または正常(NORMAL)かを判定してください。

### 判定対象デバイス一覧
{json.dumps(devices_info, ensure_ascii=False, indent=2)}

### 判定ルール
- "冗長が効いている（サービス継続）"と判断できる限り、CRITICALにしない
- サービス断（停止）が強く示唆される場合のみ CRITICAL
- 各デバイスごとに独立して判定

### 出力フォーマット
以下のJSON配列形式のみを出力（Markdownコードブロック不要）:
[
  {{"device_id": "...", "status": "NORMAL|WARNING|CRITICAL", "reason": "判定理由", "impact_type": "NONE|DEGRADED|REDUNDANCY_LOST|OUTAGE|UNKNOWN"}}
]
"""

        # トークン数チェック
        if not check_input_limit(prompt):
            logger.warning("Prompt too long, splitting batch...")
            # バッチを分割して再帰呼び出し
            mid = len(devices_alerts) // 2
            items = list(devices_alerts.items())
            results = {}
            results.update(self._analyze_batch_with_llm(dict(items[:mid])))
            results.update(self._analyze_batch_with_llm(dict(items[mid:])))
            return results

        try:
            # レート制限待機
            if not self._rate_limiter.wait_for_slot():
                raise RuntimeError("Rate limit exceeded")
            
            self._rate_limiter.record_request()
            
            response = self.model.generate_content(
                prompt, 
                generation_config={"response_mime_type": "application/json"}
            )
            response_text = response.text.strip()
            
            # JSONパース
            if response_text.startswith("```"):
                response_text = re.sub(r'^```\w*\n?', '', response_text)
                response_text = re.sub(r'\n?```$', '', response_text)
            
            result_list = json.loads(response_text)
            
            # 結果をマッピング
            results = {}
            for item in result_list:
                dev_id = item.get("device_id", "")
                if dev_id not in devices_alerts:
                    continue
                
                status_str = str(item.get("status", "WARNING")).upper()
                if status_str in ["GREEN", "NORMAL"]:
                    health_status = HealthStatus.NORMAL
                elif status_str in ["YELLOW", "WARNING"]:
                    health_status = HealthStatus.WARNING
                else:
                    health_status = HealthStatus.CRITICAL
                
                results[dev_id] = AnalysisResult(
                    device_id=dev_id,
                    status=health_status,
                    reason=item.get("reason", "AI provided no reason"),
                    impact_type=item.get("impact_type", "UNKNOWN"),
                    confidence=0.7
                )
            
            # 欠落デバイスの処理
            for dev_id in devices_alerts:
                if dev_id not in results:
                    results[dev_id] = AnalysisResult(
                        device_id=dev_id,
                        status=HealthStatus.WARNING,
                        reason="No analysis result from AI",
                        impact_type="UNKNOWN",
                        confidence=0.3
                    )
            
            # キャッシュ保存
            self._rate_limiter.set_cache(cache_key, results)
            
            logger.info(f"Batch analysis completed for {len(results)} devices")
            return results

        except Exception as e:
            logger.error(f"Batch AI Inference Error: {e}")
            return {
                dev_id: AnalysisResult(
                    device_id=dev_id,
                    status=HealthStatus.WARNING,
                    reason=f"AI Analysis Failed: {str(e)}",
                    impact_type="AI_ERROR",
                    confidence=0.2
                )
                for dev_id in devices_alerts
            }

    # ==========================================================
    # Silent failure inference
    # ==========================================================
    def _is_connection_loss(self, msg: str) -> bool:
        msg_l = msg.lower()
        return (
            "connection lost" in msg_l
            or "link down" in msg_l
            or "port down" in msg_l
            or "unreachable" in msg_l
        )

    def _detect_silent_failures(self, msg_map: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
        """親自身にアラームが無いのに、配下の複数子がConnection Lostを出しているなら親を疑う"""
        suspects: Dict[str, Dict[str, Any]] = {}

        for parent_id, children in self.children_map.items():
            if not children or parent_id in msg_map:
                continue

            affected = [c for c in children if any(self._is_connection_loss(m) for m in msg_map.get(c, []))]
            if not affected:
                continue

            total = len(children)
            ratio = len(affected) / max(total, 1)

            if len(affected) >= self.SILENT_MIN_CHILDREN and ratio >= self.SILENT_RATIO:
                report = (
                    f"[Silent Failure Heuristic]\n"
                    f"- Suspected upstream device: {parent_id}\n"
                    f"- Evidence: {len(affected)}/{total} children report connection loss\n"
                    f"- Affected children: {', '.join(affected)}\n"
                    f"- Recommendation: Check uplinks, power, and management connectivity\n"
                )
                suspects[parent_id] = {
                    "evidence_count": len(affected),
                    "total_children": total,
                    "affected_children": affected,
                    "report": report,
                }

        return suspects

    # ==========================================================
    # メイン推論メソッド
    # ==========================================================
    def infer_root_cause(self, msg_map: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        ★改善版: ローカルルール優先 + バッチ処理
        
        Args:
            msg_map: {device_id: [alert_messages...], ...}
        
        Returns:
            List of inference results
        """
        if not msg_map:
            return []

        silent_suspects = self._detect_silent_failures(msg_map)
        alarmed_ids = set(msg_map.keys())
        results: List[Dict[str, Any]] = []
        
        # LLM呼び出しが必要なデバイスを収集
        llm_needed: Dict[str, List[str]] = {}

        for device_id, messages in msg_map.items():
            # サイレント疑い配下の子は被疑扱い
            parent_id = self._get_parent_id(device_id)
            if parent_id in silent_suspects and any(self._is_connection_loss(m) for m in messages):
                results.append({
                    "id": device_id,
                    "label": " / ".join(messages),
                    "prob": 0.4,
                    "type": "Network/ConnectionLost",
                    "tier": 3,
                    "reason": f"Downstream symptom under suspected silent failure parent (parent={parent_id})."
                })
                continue

            # 通常のカスケード抑制
            if any("unreachable" in m.lower() for m in messages) and parent_id in alarmed_ids:
                results.append({
                    "id": device_id,
                    "label": " / ".join(messages),
                    "prob": 0.2,
                    "type": "Network/Unreachable",
                    "tier": 3,
                    "reason": f"Downstream unreachable due to upstream alarm (parent={parent_id})."
                })
                continue

            # サイレント疑いデバイス
            if device_id in silent_suspects:
                info = silent_suspects[device_id]
                results.append({
                    "id": device_id,
                    "label": " / ".join(messages) if messages else "Silent Failure Suspected",
                    "prob": 0.8,
                    "type": "Network/SilentFailure",
                    "tier": 1,
                    "reason": f"Silent failure suspected: {info['evidence_count']}/{info['total_children']} children affected.",
                    "analyst_report": info["report"],
                    "auto_investigation": [
                        "Pull interface counters/errors (uplinks)",
                        "Check STP/MAC flaps",
                        "Ping/ARP reachability tests from upstream",
                        "Correlate syslog around incident time"
                    ]
                })
                continue

            # ★ローカルルールで判定を試みる
            local_result = self._apply_local_rules(device_id, messages)
            if local_result:
                # ローカルルールで判定成功
                if local_result.status == HealthStatus.CRITICAL:
                    prob, tier = 0.9, 1
                elif local_result.status == HealthStatus.WARNING:
                    prob, tier = 0.7, 2
                else:
                    prob, tier = 0.3, 3
                
                results.append({
                    "id": device_id,
                    "label": " / ".join(messages),
                    "prob": prob,
                    "type": local_result.impact_type,
                    "tier": tier,
                    "reason": local_result.reason
                })
                logger.debug(f"Device {device_id}: determined by local rule")
            else:
                # LLMが必要
                llm_needed[device_id] = messages

        # ★バッチ処理でLLM呼び出し
        if llm_needed:
            logger.info(f"Calling LLM for {len(llm_needed)} devices in batch")
            
            # バッチサイズごとに分割
            items = list(llm_needed.items())
            for i in range(0, len(items), MAX_BATCH_SIZE):
                batch = dict(items[i:i + MAX_BATCH_SIZE])
                batch_results = self._analyze_batch_with_llm(batch)
                
                for dev_id, analysis in batch_results.items():
                    messages = llm_needed[dev_id]
                    
                    if analysis.status == HealthStatus.CRITICAL:
                        prob, tier = 0.9, 1
                    elif analysis.status == HealthStatus.WARNING:
                        prob, tier = 0.7, 2
                    else:
                        prob, tier = 0.3, 3
                    
                    results.append({
                        "id": dev_id,
                        "label": " / ".join(messages),
                        "prob": prob,
                        "type": analysis.impact_type,
                        "tier": tier,
                        "reason": analysis.reason
                    })

        results.sort(key=lambda x: x["prob"], reverse=True)
        return results

    # 後方互換性のためのエイリアス
    def analyze_redundancy_depth(self, device_id: str, alerts: List[str]) -> Dict[str, Any]:
        """
        後方互換性のためのラッパー
        ★注意: 個別呼び出しは非推奨。infer_root_cause() を使用してください。
        """
        local_result = self._apply_local_rules(device_id, alerts)
        if local_result:
            return {
                "status": local_result.status,
                "reason": local_result.reason,
                "impact_type": local_result.impact_type
            }
        
        # 単一デバイスでもバッチ処理を使用
        batch_results = self._analyze_batch_with_llm({device_id: alerts})
        if device_id in batch_results:
            analysis = batch_results[device_id]
            return {
                "status": analysis.status,
                "reason": analysis.reason,
                "impact_type": analysis.impact_type
            }
        
        return {
            "status": HealthStatus.WARNING,
            "reason": "Analysis failed",
            "impact_type": "UNKNOWN"
        }

    def analyze(self, alarms) -> List[Dict[str, Any]]:
        """
        後方互換性のためのラッパー
        
        Args:
            alarms: Alarmオブジェクトのリスト（device_id, message属性を持つ）
        
        Returns:
            分析結果のリスト
        """
        # alarmsからmsg_mapを構築
        msg_map: Dict[str, List[str]] = {}
        for alarm in alarms:
            device_id = getattr(alarm, 'device_id', None)
            message = getattr(alarm, 'message', None)
            if device_id and message:
                if device_id not in msg_map:
                    msg_map[device_id] = []
                msg_map[device_id].append(message)
        
        # infer_root_causeを呼び出し
        return self.infer_root_cause(msg_map)
