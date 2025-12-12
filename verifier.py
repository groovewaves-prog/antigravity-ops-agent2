"""
Google Antigravity AIOps Agent - Verification Module
ハルシネーション防止のための高速・高精度な検証モジュール
"""
import re

# =====================================================
# パターンキャッシュ（遅延初期化クラス）
# =====================================================

class _PatternCache:
    """正規表現パターンを保持するシングルトン"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._compile_patterns()
            self._initialized = True
    
    def _compile_patterns(self):
        # Pingパターン
        self.ping_stats = re.compile(
            r'(?:(\d+)\s+packets?\s+transmitted.*?(\d+)\s+received)|'
            r'(?:success\s+rate\s+is\s+(\d+)\s*percent)|'
            r'(?:(\d+)%\s+packet\s+loss)',
            re.I
        )
        self.ping_fail_fast = re.compile(r'(100%\s+packet\s+loss|unreachable|timed?\s*out|0\s+received)', re.I)
        
        # インターフェース
        self.admin_down = re.compile(r'administratively\s+down', re.I)
        self.if_status = re.compile(
            r'(?:line\s+protocol\s+is\s+(up|down))|'
            r'(?:interface\s+is\s+(up|down))|'
            r'(?:(err-disabled|notconnect))',
            re.I
        )
        
        # ハードウェア
        self.hw_check = re.compile(
            r'(fan|power|psu|temp|environment|sensor).*?'
            r'(fail(ed|ure)?|fault(y)?|critical|ok|good|normal|warn(ing)?)',
            re.I | re.DOTALL
        )

_cache = None

def _get_cache():
    global _cache
    if _cache is None:
        _cache = _PatternCache()
    return _cache

# =====================================================
# 検証ロジック
# =====================================================

def verify_log_content(log_text: str) -> dict:
    """
    ログテキストから客観的事実を抽出する
    """
    # 空の結果定義
    result = {
        "ping_status": "Unknown", "ping_confidence": 0.0, "ping_evidence": "",
        "interface_status": "Unknown", "interface_confidence": 0.0, "interface_evidence": "",
        "hardware_status": "Unknown", "hardware_confidence": 0.0, "hardware_evidence": "",
        "error_keywords": "None", "error_severity": 0.0,
        "conflicts_detected": [], "overall_confidence": 0.0
    }

    if not log_text:
        return result
    
    cache = _get_cache()
    text_lower = log_text.lower()
    
    # 1. Ping検証
    _fast_verify_ping(text_lower, cache, result)
    
    # 2. Interface検証
    _fast_verify_interface(text_lower, cache, result)
    
    # 3. Hardware検証
    if any(kw in text_lower for kw in ['fan', 'power', 'psu', 'temp', 'environment']):
        _fast_verify_hardware(text_lower, cache, result)
    
    # 4. エラーキーワード
    _fast_verify_errors(text_lower, result)
    
    # 5. 矛盾検知
    _detect_simple_conflicts(result)
    
    # 全体信頼度計算
    confidences = [
        result.get("ping_confidence", 0),
        result.get("interface_confidence", 0),
        result.get("hardware_confidence", 0)
    ]
    result["overall_confidence"] = max(confidences) if any(confidences) else 0.0
    
    return result

def _fast_verify_ping(text: str, cache, result: dict):
    if 'ping' not in text and 'icmp' not in text:
        return
    
    fail_match = cache.ping_fail_fast.search(text)
    if fail_match:
        result.update({
            "ping_status": "CRITICAL",
            "ping_confidence": 0.9,
            "ping_evidence": f"Failure: {fail_match.group(1)}"
        })
        return
    
    stats_match = cache.ping_stats.search(text)
    if stats_match:
        groups = stats_match.groups()
        success_rate = None
        try:
            if groups[0] and groups[1]:
                sent, received = int(groups[0]), int(groups[1])
                success_rate = (received / sent * 100) if sent > 0 else 0
            elif groups[2]:
                success_rate = int(groups[2])
            elif groups[3]:
                success_rate = 100 - int(groups[3])
            
            if success_rate is not None:
                if success_rate >= 80:
                    status, conf = "OK", 0.9
                elif success_rate >= 50:
                    status, conf = "WARNING", 0.7
                else:
                    status, conf = "CRITICAL", 0.8
                
                result.update({
                    "ping_status": status,
                    "ping_confidence": conf,
                    "ping_evidence": f"Success rate: {success_rate:.0f}%"
                })
        except (ValueError, ZeroDivisionError):
            pass

def _fast_verify_interface(text: str, cache, result: dict):
    if cache.admin_down.search(text):
        result.update({
            "interface_status": "INFO",
            "interface_confidence": 0.9,
            "interface_evidence": "Admin down (intentional)"
        })
        return
    
    status_match = cache.if_status.findall(text)
    if not status_match:
        return
    
    down_count = sum(1 for m in status_match if 'down' in str(m).lower() or 'disabled' in str(m).lower())
    up_count = sum(1 for m in status_match if 'up' in str(m).lower())
    
    if down_count > up_count:
        result.update({
            "interface_status": "CRITICAL",
            "interface_confidence": 0.9,
            "interface_evidence": f"Link DOWN detected ({down_count} interfaces)"
        })
    elif up_count > down_count:
        result.update({
            "interface_status": "OK",
            "interface_confidence": 0.8,
            "interface_evidence": f"Link UP ({up_count} interfaces)"
        })
    else:
        result.update({
            "interface_status": "WARNING",
            "interface_confidence": 0.5,
            "interface_evidence": "Mixed states"
        })

def _fast_verify_hardware(text: str, cache, result: dict):
    hw_matches = cache.hw_check.findall(text)
    if not hw_matches:
        return
    
    critical_count = sum(1 for m in hw_matches if any(k in str(m).lower() for k in ['fail', 'fault', 'critical']))
    ok_count = sum(1 for m in hw_matches if any(k in str(m).lower() for k in ['ok', 'good', 'normal']))
    warning_count = sum(1 for m in hw_matches if 'warn' in str(m).lower())
    
    if critical_count > 0:
        result.update({
            "hardware_status": "CRITICAL",
            "hardware_confidence": 0.9,
            "hardware_evidence": f"HW failure detected ({critical_count} issues)"
        })
    elif warning_count > 0:
        result.update({
            "hardware_status": "WARNING",
            "hardware_confidence": 0.8,
            "hardware_evidence": f"HW warning ({warning_count} issues)"
        })
    elif ok_count > 0:
        result.update({
            "hardware_status": "OK",
            "hardware_confidence": 0.8,
            "hardware_evidence": f"HW OK ({ok_count} components)"
        })

def _fast_verify_errors(text: str, result: dict):
    critical_keywords = ['crash', 'panic', 'fatal', 'severe']
    error_keywords = ['error', 'fail', 'exception', 'denied']
    
    found_critical = [k for k in critical_keywords if k in text]
    found_errors = [k for k in error_keywords if k in text and k not in found_critical]
    
    if found_critical:
        result.update({
            "error_keywords": f"Critical: {', '.join(found_critical[:3])}",
            "error_severity": 0.9
        })
    elif found_errors:
        result.update({
            "error_keywords": f"Errors: {', '.join(found_errors[:3])}",
            "error_severity": 0.7
        })

def _detect_simple_conflicts(result: dict):
    conflicts = []
    ping_ok = result.get("ping_status") == "OK"
    if_down = result.get("interface_status") == "CRITICAL"
    
    if ping_ok and if_down:
        conflicts.append("矛盾検知: Ping疎通は成功していますが、I/Fダウンが検出されています")
    
    result["conflicts_detected"] = conflicts

# =====================================================
# レポートフォーマット関数（必須）
# =====================================================

def format_verification_report(facts: dict) -> str:
    """検証結果を整形して返す"""
    overall_conf = facts.get('overall_confidence', 0)
    confidence_level = "高" if overall_conf >= 0.8 else "中" if overall_conf >= 0.5 else "低"
    
    report = f"""
【システム自動検証結果 (Ground Truth)】
※AIの推論はこの客観的事実と矛盾してはならない

◆ 総合信頼度: {confidence_level} ({overall_conf:.0%})

◆ 疎通: {facts.get('ping_status', 'N/A')} (信頼度: {facts.get('ping_confidence', 0):.0%})
  → {facts.get('ping_evidence', 'N/A')}

◆ インターフェース: {facts.get('interface_status', 'N/A')} (信頼度: {facts.get('interface_confidence', 0):.0%})
  → {facts.get('interface_evidence', 'N/A')}

◆ ハードウェア: {facts.get('hardware_status', 'N/A')} (信頼度: {facts.get('hardware_confidence', 0):.0%})
  → {facts.get('hardware_evidence', 'N/A')}

◆ エラー: {facts.get('error_keywords', 'N/A')}
"""
    
    if facts.get('conflicts_detected'):
        report += f"\n⚠️ **矛盾検知**: {'; '.join(facts['conflicts_detected'])}\n"
    
    return report
