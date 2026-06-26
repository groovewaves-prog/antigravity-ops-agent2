import streamlit as st
import graphviz
import os
import time
import google.generativeai as genai
import json
import re
import pandas as pd
from google.api_core import exceptions as google_exceptions

# モジュール群のインポート
from data import TOPOLOGY
from model_config import GEMMA_MODEL
from logic import CausalInferenceEngine, Alarm, simulate_cascade_failure
from network_ops import (
    run_diagnostic_simulation, 
    generate_remediation_commands, 
    generate_analyst_report,
    generate_analyst_report_streaming,           # ★新規: ストリーミング版
    generate_remediation_commands_streaming,    # ★新規: ストリーミング版
    compute_cache_hash,                         # ★新規: キャッシュハッシュ
    predict_initial_symptoms, 
    generate_fake_log_by_ai,
    run_remediation_parallel_v2,
    RemediationEnvironment,
    RemediationResult
)
from verifier import verify_log_content, format_verification_report
from inference_engine import LogicalRCA
from rate_limiter import GlobalRateLimiter, RateLimitConfig
import hashlib

# --- ページ設定 ---
st.set_page_config(page_title="Antigravity Autonomous", page_icon="⚡", layout="wide")

# =====================================================
# レートリミッター初期化
# =====================================================
@st.cache_resource
def get_rate_limiter():
    """レートリミッターのシングルトンインスタンスを取得"""
    return GlobalRateLimiter(RateLimitConfig(
        rpm=30,
        rpd=14400,
        safety_margin=0.9
    ))

rate_limiter = get_rate_limiter()

# ==========================================
# 関数定義
# ==========================================
def find_target_node_id(topology, node_type=None, layer=None, keyword=None):
    """トポロジーから条件に合うノードIDを検索"""
    for node_id, node in topology.items():
        if node_type and node.type != node_type: continue
        if layer and node.layer != layer: continue
        if keyword:
            hit = False
            if keyword in node_id: hit = True
            for v in node.metadata.values():
                if isinstance(v, str) and keyword in v: hit = True
            if not hit: continue
        return node_id
    return None

def load_config_by_id(device_id):
    """configsフォルダから設定ファイルを読み込む"""
    possible_paths = [f"configs/{device_id}.txt", f"{device_id}.txt"]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
    return "Config file not found."

def generate_content_with_retry(model, prompt, stream=True, retries=3):
    """503エラー対策のリトライ付き生成関数（レートリミッター統合）"""
    for i in range(retries):
        try:
            # レート制限チェック
            if not rate_limiter.wait_for_slot(timeout=60):
                raise RuntimeError("Rate limit timeout")
            rate_limiter.record_request()
            return model.generate_content(prompt, stream=stream)
        except google_exceptions.ServiceUnavailable:
            if i == retries - 1: raise
            time.sleep(2 * (i + 1))
        except Exception as e:
            if '429' in str(e) or 'rate' in str(e).lower():
                if i == retries - 1: raise
                time.sleep(5 * (i + 1))
            else:
                raise
    return None


def _hash_text(text: str) -> str:
    """テキストのハッシュ値を計算"""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _pick_first(mapping: dict, keys: list[str], default: str = "") -> str:
    """Return the first non-empty value for the given keys from mapping (stringify scalars)."""
    for k in keys:
        try:
            v = mapping.get(k, None)
        except Exception:
            v = None
        if v is None:
            continue
        if isinstance(v, (int, float, bool)):
            s = str(v)
            if s:
                return s
        elif isinstance(v, str):
            if v.strip():
                return v.strip()
        else:
            # for non-string, try json
            try:
                s = json.dumps(v, ensure_ascii=False)
                if s and s != "null":
                    return s
            except Exception:
                continue
    return default


def _build_ci_context_for_chat(target_node_id: str) -> dict:
    """チャット用のCIコンテキストを最小限で構築します（CI/Configをフル活用、キー揺れを吸収）。"""
    node = TOPOLOGY.get(target_node_id) if target_node_id else None
    md = (getattr(node, "metadata", None) or {}) if node else {}

    ci = {
        "device_id": target_node_id or "",
        "hostname": _pick_first(md, ["hostname", "host", "name"], default=(target_node_id or "")),
        "vendor": _pick_first(md, ["vendor", "manufacturer", "maker", "brand"], default=""),
        "os": _pick_first(md, ["os", "platform", "os_name", "software", "sw"], default=""),
        "model": _pick_first(md, ["model", "hw_model", "product", "sku"], default=""),
        "role": _pick_first(md, ["role", "type", "device_role"], default=""),
        "layer": _pick_first(md, ["layer", "level", "network_layer"], default=""),
        "site": _pick_first(md, ["site", "dc", "datacenter", "location"], default=""),
        "tenant": _pick_first(md, ["tenant", "customer", "org", "company"], default=""),
        "mgmt_ip": _pick_first(md, ["mgmt_ip", "management_ip", "management", "oob_ip"], default=""),
        "interfaces": md.get("interfaces", ""),
    }

    # Config は長いので抜粋（存在すれば最大1500文字）
    try:
        conf = load_config_by_id(target_node_id) if target_node_id else ""
        if conf:
            ci["config_excerpt"] = conf[:1500]
    except Exception:
        pass

    return ci


def _safe_chunk_text(chunk) -> str:
    """google.generativeai の stream chunk から安全にテキストを取り出します。"""
    # chunk.text は ValueError になり得る
    try:
        t = getattr(chunk, "text", "")
        if t:
            return t
    except Exception:
        pass

    # candidates -> content -> parts から拾う
    try:
        cands = getattr(chunk, "candidates", None) or []
        if not cands:
            return ""
        content = getattr(cands[0], "content", None)
        parts = getattr(content, "parts", None) or []
        out = []
        for p in parts:
            tx = getattr(p, "text", "")
            if tx:
                out.append(tx)
        return "".join(out)
    except Exception:
        return ""




def run_diagnostic_simulation_no_llm(selected_scenario, target_node_obj):
    """LLMを呼ばない疑似診断（503/コスト対策）。UXは維持しつつ、材料を増やすためのログを生成します。
    重要: 「修復実行(Execute)」で復旧成功した後は、同一シナリオに限り成功側の疑似ログを返します。
    """
    device_id = getattr(target_node_obj, "id", "UNKNOWN") if target_node_obj else "UNKNOWN"
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"[PROBE] ts={ts}",
        f"[PROBE] scenario={selected_scenario}",
        f"[PROBE] target_device={device_id}",
        "",
    ]

    # 復旧成功フラグ（デモ用）
    recovered_devices = st.session_state.get("recovered_devices") or {}
    recovered_map = st.session_state.get("recovered_scenario_map") or {}

    if recovered_devices.get(device_id) and recovered_map.get(device_id) == selected_scenario:
        # "復旧後"の疑似ログ（成功）
        if "FW" in selected_scenario:
            lines += [
                "show chassis cluster status",
                "Redundancy group 0: healthy",
                "control link: up",
                "fabric link: up",
            ]
        elif "WAN" in selected_scenario or "WAN全回線断" in selected_scenario:
            lines += [
                "show ip interface brief",
                "GigabitEthernet0/0 up up",
                "show ip bgp summary",
                "Neighbor 203.0.113.2 Established",
                "ping 203.0.113.2 repeat 5",
                "Success rate is 100 percent (5/5)",
            ]
        elif "L2SW" in selected_scenario:
            lines += [
                "show environment",
                "Fan: OK",
                "Temperature: OK",
                "show interface status",
                "Uplink: up",
            ]
        else:
            lines += [
                "show system alarms",
                "No active alarms",
                "ping 8.8.8.8 repeat 5",
                "Success rate is 100 percent (5/5)",
            ]

        return {
            "status": "SUCCESS",
            "sanitized_log": "\n".join(lines),
            "verification_log": "N/A",
            "device_id": device_id,
        }

    # "障害中"の疑似ログ（現状維持）
    if "WAN全回線断" in selected_scenario or "[WAN]" in selected_scenario:
        lines += [
            "show ip interface brief",
            "GigabitEthernet0/0 down down",
            "show ip bgp summary",
            "Neighbor 203.0.113.2 Idle",
            "ping 203.0.113.2 repeat 5",
            "Success rate is 0 percent (0/5)",
        ]
    elif "FW片系障害" in selected_scenario or "[FW]" in selected_scenario:
        lines += [
            "show chassis cluster status",
            "Redundancy group 0: degraded",
            "control link: down",
            "fabric link: up",
        ]
    elif "L2SW" in selected_scenario:
        lines += [
            "show environment",
            "Fan: FAIL",
            "Temperature: HIGH",
            "show interface status",
            "Uplink: flapping",
        ]
    else:
        lines += [
            "show system alarms",
            "No active alarms",
        ]

    return {
        "status": "SUCCESS",
        "sanitized_log": "\n".join(lines),
        "verification_log": "N/A",
        "device_id": device_id,
    }


def _hash_text(text: str) -> str:
    import hashlib
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]

def _extract_first_codeblock_after_heading(markdown_text: str, heading_keyword: str) -> str:
    """Extract the first fenced code block (``` ... ```) that appears *after* a heading containing heading_keyword.
    - Returns code content without fences.
    - If not found, returns empty string.
    This is intentionally simple and robust to avoid complex parsing / IF sprawl.
    """
    if not markdown_text or not heading_keyword:
        return ""
    # Find the heading position (supports '#', '##', etc. and also plain text headings)
    idx = markdown_text.find(heading_keyword)
    if idx < 0:
        return ""
    tail = markdown_text[idx:]
    # Find first fenced code block after the heading
    m = re.search(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)\n```", tail, flags=re.DOTALL)
    if not m:
        return ""
    return (m.group(1) or "").strip()
def render_topology(alarms, root_cause_candidates):
    """トポロジー図の描画 (AI判定結果を反映)"""
    graph = graphviz.Digraph()
    graph.attr(rankdir='TB')
    graph.attr('node', shape='box', style='rounded,filled', fontname='Helvetica')
    
    alarm_map = {a.device_id: a for a in alarms}
    alarmed_ids = set(alarm_map.keys())
    
    root_cause_ids = {c['id'] for c in root_cause_candidates if c['prob'] > 0.6}
    
    # AI判定結果のマッピング
    node_status_map = {c['id']: c['type'] for c in root_cause_candidates}
    
    for node_id, node in TOPOLOGY.items():
        color = "#e8f5e9"
        penwidth = "1"
        fontcolor = "black"
        label = f"{node_id}\n({node.type})"
        
        red_type = node.metadata.get("redundancy_type")
        if red_type: label += f"\n[{red_type} Redundancy]"
        vendor = node.metadata.get("vendor")
        if vendor: label += f"\n[{vendor}]"

        status_type = node_status_map.get(node_id, "Normal")
        
        if "Hardware/Physical" in status_type or "Critical" in status_type or "Silent" in status_type:
            color = "#ffcdd2" 
            penwidth = "3"
            label += "\n[ROOT CAUSE]"
        elif "Network/Unreachable" in status_type or "Network/Secondary" in status_type:
            color = "#cfd8dc" 
            fontcolor = "#546e7a"
            label += "\n[Unreachable]"
        elif node_id in alarmed_ids:
            color = "#fff9c4" 
        
        graph.node(node_id, label=label, fillcolor=color, color='black', penwidth=penwidth, fontcolor=fontcolor)
    
    for node_id, node in TOPOLOGY.items():
        if node.parent_id:
            graph.edge(node.parent_id, node_id)
            parent_node = TOPOLOGY.get(node.parent_id)
            if parent_node and parent_node.redundancy_group:
                partners = [n.id for n in TOPOLOGY.values() 
                           if n.redundancy_group == parent_node.redundancy_group and n.id != parent_node.id]
                for partner_id in partners:
                    graph.edge(partner_id, node_id)
    return graph

# --- UI構築 ---
st.title("⚡ Antigravity Autonomous Agent")

api_key = None
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = os.environ.get("GOOGLE_API_KEY")

# --- サイドバー ---
with st.sidebar:
    st.header("⚡ Scenario Controller")
    SCENARIO_MAP = {
        "基本・広域障害": ["正常稼働", "1. WAN全回線断", "2. FW片系障害", "3. L2SWサイレント障害"],
        "WAN Router": ["4. [WAN] 電源障害：片系", "5. [WAN] 電源障害：両系", "6. [WAN] BGPルートフラッピング", "7. [WAN] FAN故障", "8. [WAN] メモリリーク"],
        "Firewall (Juniper)": ["9. [FW] 電源障害：片系", "10. [FW] 電源障害：両系", "11. [FW] FAN故障", "12. [FW] メモリリーク"],
        "L2 Switch": ["13. [L2SW] 電源障害：片系", "14. [L2SW] 電源障害：両系", "15. [L2SW] FAN故障", "16. [L2SW] メモリリーク"],
        "複合・その他": ["17. [WAN] 複合障害：電源＆FAN", "18. [Complex] 同時多発：FW & AP", "99. [Live] Cisco実機診断"]
    }
    selected_category = st.selectbox("対象カテゴリ:", list(SCENARIO_MAP.keys()))
    selected_scenario = st.radio("発生シナリオ:", SCENARIO_MAP[selected_category])
    st.markdown("---")
    if api_key: 
        st.success("API Connected")
        # レート制限状況の表示
        stats = rate_limiter.get_stats()
        st.caption(f"📊 API: {stats['requests_last_minute']}/{stats['rpm_limit']} RPM")
    else:
        st.warning("API Key Missing")
        user_key = st.text_input("Google API Key", type="password")
        if user_key: api_key = user_key

# --- セッション管理 ---
if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = "正常稼働"

# 変数初期化
for key in ["live_result", "messages", "chat_session", "trigger_analysis", "verification_result", "generated_report", "verification_log", "last_report_cand_id", "logic_engine", "recovered_devices", "recovered_scenario_map", "balloons_shown", "report_cache"]:
    if key not in st.session_state:
        if key == "messages":
            st.session_state[key] = []
        elif key in ["trigger_analysis", "balloons_shown"]:
            st.session_state[key] = False
        elif key == "report_cache":
            st.session_state[key] = {}
        else:
            st.session_state[key] = None


# 復旧状態（デモ用）
if "recovered_devices" not in st.session_state:
    st.session_state.recovered_devices = {}
if "recovered_scenario_map" not in st.session_state:
    st.session_state.recovered_scenario_map = {}

# ★パフォーマンス改善：グローバルキャッシュの初期化
if "global_cache" not in st.session_state:
    st.session_state.global_cache = {}

GLOBAL_CACHE = st.session_state.global_cache

# エンジン初期化
if not st.session_state.logic_engine:
    st.session_state.logic_engine = LogicalRCA(TOPOLOGY)

# シナリオ切り替え時のリセット
if st.session_state.current_scenario != selected_scenario:
    st.session_state.current_scenario = selected_scenario
    # シナリオ変更時は復旧フラグもクリア（未修復なのにOKになるのを防ぐ）
    st.session_state.recovered_devices = {}
    st.session_state.recovered_scenario_map = {}
    st.session_state.messages = []      
    st.session_state.chat_session = None 
    st.session_state.live_result = None 
    st.session_state.trigger_analysis = False
    st.session_state.verification_result = None
    st.session_state.generated_report = None
    st.session_state.verification_log = None 
    st.session_state.last_report_cand_id = None
    st.session_state.balloons_shown = False  # バルーンフラグもリセット
    if "remediation_plan" in st.session_state: del st.session_state.remediation_plan
    st.rerun()

# ==========================================
# メインロジック
# ==========================================
alarms = []
target_device_id = None
root_severity = "CRITICAL"
is_live_mode = False

# 1. アラーム生成ロジック
if "Live" in selected_scenario: is_live_mode = True
elif "WAN全回線断" in selected_scenario:
    target_device_id = find_target_node_id(TOPOLOGY, node_type="ROUTER")
    if target_device_id: alarms = simulate_cascade_failure(target_device_id, TOPOLOGY)
elif "FW片系障害" in selected_scenario:
    target_device_id = find_target_node_id(TOPOLOGY, node_type="FIREWALL")
    if target_device_id:
        alarms = [Alarm(target_device_id, "Heartbeat Loss", "WARNING")]
        root_severity = "WARNING"

elif "L2SWサイレント障害" in selected_scenario:
    target_device_id = "L2_SW_01"
    if target_device_id not in TOPOLOGY:
        target_device_id = find_target_node_id(TOPOLOGY, keyword="L2_SW")
    if target_device_id and target_device_id in TOPOLOGY:
        child_nodes = [nid for nid, n in TOPOLOGY.items() if n.parent_id == target_device_id]
        alarms = [Alarm(child, "Connection Lost", "CRITICAL") for child in child_nodes]
    else:
        st.error("Error: L2 Switch definition not found")

elif "複合障害" in selected_scenario:
    target_device_id = find_target_node_id(TOPOLOGY, node_type="ROUTER")
    if target_device_id:
        alarms = [
            Alarm(target_device_id, "Power Supply 1 Failed", "CRITICAL"),
            Alarm(target_device_id, "Fan Fail", "WARNING")
        ]
elif "同時多発" in selected_scenario:
    fw_node = find_target_node_id(TOPOLOGY, node_type="FIREWALL")
    ap_node = find_target_node_id(TOPOLOGY, node_type="ACCESS_POINT")
    alarms = []
    if fw_node: alarms.append(Alarm(fw_node, "Heartbeat Loss", "WARNING"))
    if ap_node: alarms.append(Alarm(ap_node, "Connection Lost", "CRITICAL"))
    target_device_id = fw_node 
else:
    if "[WAN]" in selected_scenario: target_device_id = find_target_node_id(TOPOLOGY, node_type="ROUTER")
    elif "[FW]" in selected_scenario: target_device_id = find_target_node_id(TOPOLOGY, node_type="FIREWALL")
    elif "[L2SW]" in selected_scenario: target_device_id = find_target_node_id(TOPOLOGY, node_type="SWITCH", layer=4)

    if target_device_id:
        if "電源障害：片系" in selected_scenario:
            alarms = [Alarm(target_device_id, "Power Supply 1 Failed", "WARNING")]
            root_severity = "WARNING"
        elif "電源障害：両系" in selected_scenario:
            if "FW" in target_device_id:
                alarms = [Alarm(target_device_id, "Power Supply: Dual Loss (Device Down)", "CRITICAL")]
            else:
                alarms = simulate_cascade_failure(target_device_id, TOPOLOGY, "Power Supply: Dual Loss (Device Down)")
        elif "BGP" in selected_scenario:
            alarms = [Alarm(target_device_id, "BGP Flapping", "WARNING")]
            root_severity = "WARNING"
        elif "FAN" in selected_scenario:
            alarms = [Alarm(target_device_id, "Fan Fail", "WARNING")]
            root_severity = "WARNING"
        elif "メモリ" in selected_scenario:
            alarms = [Alarm(target_device_id, "Memory High", "WARNING")]
            root_severity = "WARNING"

# 2. 推論エンジンによる分析
analysis_results = st.session_state.logic_engine.analyze(alarms)

# 3. コックピット表示
selected_incident_candidate = None

st.markdown("### 🛡️ AIOps インシデント・コックピット")
col1, col2, col3 = st.columns(3)
with col1: st.metric("📉 ノイズ削減率", "98.5%", "高効率稼働中")
with col2: st.metric("📨 処理アラーム数", f"{len(alarms) * 15 if alarms else 0}件", "抑制済")
with col3: st.metric("🚨 要対応インシデント", f"{len([c for c in analysis_results if c['prob'] > 0.6])}件", "対処が必要")
st.markdown("---")

df_data = []
# ★修正: スライス制限を撤廃 (全件表示)
# 階層ロジックにより、重要なもの(Tier高)が先頭に来るため、大量にあっても問題ない
for rank, cand in enumerate(analysis_results, 1):
    status = "⚪ 監視中"
    action = "👁️ 静観"
    
    if cand['prob'] > 0.8:
        status = "🔴 危険 (根本原因)"
        action = "🚀 自動修復が可能"
    elif cand['prob'] > 0.6:
        status = "🟡 警告 (被疑箇所)"
        action = "🔍 詳細調査を推奨"
    
    if "Network/Unreachable" in cand['type'] or "Network/Secondary" in cand['type']:
        status = "⚫ 応答なし (上位障害)"
        action = "⛔ 対応不要 (上位復旧待ち)"

    candidate_text = f"デバイス: {cand['id']} / 原因: {cand['label']}"
    if cand.get('verification_log'):
        candidate_text += " [🔍 Active Probe: 応答なし]"
    
    # デバッグ用にTierを表示（本番では消しても良い）
    # candidate_text += f" (Tier: {cand.get('tier')})"

    df_data.append({
        "順位": rank,
        "ステータス": status,
        "根本原因候補": candidate_text,
        "リスクスコア": cand['prob'],
        "推奨アクション": action,
        "ID": cand['id'],
        "Type": cand['type']
    })

df = pd.DataFrame(df_data)
st.info("💡 障害発生状況は障害レポートから確認できます。障害レポート作成後に復旧プランを作成できます。")

event = st.dataframe(
    df,
    column_order=["順位", "ステータス", "根本原因候補", "リスクスコア", "推奨アクション"],
    column_config={
        "リスクスコア": st.column_config.ProgressColumn("リスクスコア (0-1.0)", format="%.2f", min_value=0, max_value=1),
    },
    use_container_width=True,
    hide_index=True,
    selection_mode="single-row",
    on_select="rerun"
)

if len(event.selection.rows) > 0:
    idx = event.selection.rows[0]
    sel_row = df.iloc[idx]
    for res in analysis_results:
        if res['id'] == sel_row['ID'] and res['type'] == sel_row['Type']:
            selected_incident_candidate = res
            break
else:
    selected_incident_candidate = analysis_results[0] if analysis_results else None


# 4. 画面分割
col_map, col_chat = st.columns([1.2, 1])

# === 左カラム: トポロジーと診断 ===
with col_map:
    st.subheader("🌐 Network Topology")
    
    current_root_node = None
    current_severity = "WARNING"
    
    if selected_incident_candidate and selected_incident_candidate["prob"] > 0.6:
        current_root_node = TOPOLOGY.get(selected_incident_candidate["id"])
        if "Hardware/Physical" in selected_incident_candidate["type"] or "Critical" in selected_incident_candidate["type"] or "Silent" in selected_incident_candidate["type"]:
            current_severity = "CRITICAL"
        else:
            current_severity = "WARNING"

    elif target_device_id:
        current_root_node = TOPOLOGY.get(target_device_id)
        current_severity = root_severity

    st.graphviz_chart(render_topology(alarms, analysis_results), use_container_width=True)

    st.markdown("---")
    st.subheader("🛠️ Auto-Diagnostics")
    
    if st.button("🚀 診断実行 (Run Diagnostics)", type="primary"):
        if not api_key:
            st.error("API Key Required")
        else:
            with st.status("Agent Operating...", expanded=True) as status:
                st.write("🔌 Connecting to device...")
                target_node_obj = TOPOLOGY.get(target_device_id) if target_device_id else None
                is_live_mode = bool(st.session_state.get('api_connected')) and ('[Live]' in selected_scenario or 'Live' in selected_scenario)
                
                res = run_diagnostic_simulation(selected_scenario, target_node_obj, api_key) if is_live_mode else run_diagnostic_simulation_no_llm(selected_scenario, target_node_obj)
                st.session_state.live_result = res
                
                if res["status"] == "SUCCESS":
                    st.write("✅ Log Acquired & Sanitized.")
                    status.update(label="Diagnostics Complete!", state="complete", expanded=False)
                    log_content = res.get('sanitized_log', "")
                    verification = verify_log_content(log_content)
                    st.session_state.verification_result = verification
                    st.session_state.trigger_analysis = True
                elif res["status"] == "SKIPPED":
                    status.update(label="No Action Required", state="complete")
                else:
                    st.write("❌ Connection Failed.")
                    status.update(label="Diagnostics Failed", state="error")
            st.rerun()

    if st.session_state.live_result:
        res = st.session_state.live_result
        if res["status"] == "SUCCESS":
            st.markdown("#### 📄 Diagnostic Results")
            with st.container(border=True):
                if selected_incident_candidate and selected_incident_candidate.get("verification_log"):
                    st.caption("🤖 Active Probe / Verification Log")
                    st.code(selected_incident_candidate["verification_log"], language="text")
                    st.divider()

                if st.session_state.verification_result:
                    v = st.session_state.verification_result
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Ping Status", v.get('ping_status'))
                    c2.metric("Interface", v.get('interface_status'))
                    c3.metric("Hardware", v.get('hardware_status'))
                
                st.divider()
                st.caption("🔒 Raw Logs (Sanitized)")
                st.code(res["sanitized_log"], language="text")
        elif res["status"] == "ERROR":
            st.error(f"診断エラー: {res.get('error')}")

# === 右カラム: 分析レポート ===
with col_chat:
    st.subheader("📝 AI Analyst Report")
    
    if selected_incident_candidate:
        cand = selected_incident_candidate
        
        # --- A. 原因分析レポート (Root Cause Analysis) ---
        if "generated_report" not in st.session_state or st.session_state.generated_report is None:
            st.info(f"インシデント選択中: **{cand['id']}** ({cand['label']})")
            
            if api_key and selected_scenario != "正常稼働":
                if st.button("📝 詳細レポートを作成 (Generate Report)"):
                    
                    report_container = st.empty()
                    target_conf = load_config_by_id(cand['id'])
                    
                    verification_context = cand.get("verification_log", "特になし")
                    target_conf = load_config_by_id(cand['id'])

                    # CI/トポロジー情報
                    t_node = TOPOLOGY.get(cand["id"])
                    t_node_dict = {
    "id": getattr(t_node, "id", None),
    "type": getattr(t_node, "type", None),
    "layer": getattr(t_node, "layer", None),
    "metadata": getattr(t_node, "metadata", {}) or {},
    "parent": getattr(t_node, "parent", None),
    "children": getattr(t_node, "children", []) or [],
} if t_node else {}

                    parent_id = t_node.parent_id if t_node else None
                    children_ids = [
                        nid for nid, n in TOPOLOGY.items()
                        if getattr(n, "parent_id", None) == cand["id"]
                    ]
                    topology_context = {"node": t_node_dict, "parent_id": parent_id, "children_ids": children_ids}

                    # キャッシュキーを生成
                    cache_key_analyst = "|".join([
                        "analyst",  # Analyst専用キー
                        selected_scenario,
                        str(cand.get("id")),
                        _hash_text(json.dumps(topology_context, ensure_ascii=False, sort_keys=True)),
                        _hash_text(target_conf or ""),
                        _hash_text(verification_context or ""),
                    ])

                    if "report_cache" not in st.session_state:
                        st.session_state.report_cache = {}

                    if cache_key_analyst in st.session_state.report_cache:
                        full_text = st.session_state.report_cache[cache_key_analyst]
                        report_container.markdown(full_text)
                    else:
                        # ★パフォーマンス改善: ストリーミング対応 + エラーハンドリング強化
                        try:
                            report_container.write("🤖 AI 分析中...")
                            placeholder = report_container.empty()
                            full_text = ""
                            error_occurred = False
                            
                            # ストリーミングで段階的に取得・表示
                            try:
                                for chunk in generate_analyst_report_streaming(
                                    scenario=selected_scenario,
                                    target_node=t_node,
                                    topology_context=topology_context,
                                    target_conf=target_conf or "なし",
                                    verification_context=verification_context,
                                    api_key=api_key,
                                    max_retries=2,  # リトライ回数
                                    backoff=3       # リトライ間隔
                                ):
                                    full_text += chunk
                                    placeholder.markdown(full_text)  # 段階的に更新
                            except google_exceptions.ServiceUnavailable:
                                error_occurred = True
                                full_text += "\n\n⚠️ **API が混雑しています。生成済みレポートを表示します。**"
                                placeholder.markdown(full_text)
                            
                            if not full_text or full_text.startswith("Error"):
                                full_text = f"⚠️ 分析レポート生成に失敗しました: {full_text}"
                                placeholder.markdown(full_text)
                            
                            # 部分的でもキャッシュに保存
                            st.session_state.report_cache[cache_key_analyst] = full_text
                        except google_exceptions.ServiceUnavailable:
                            full_text = "⚠️ 現在、AIモデルが混雑しています (503 Error)。時間を置いて再度お試しください。"
                            report_container.markdown(full_text)
                        except Exception as e:
                            full_text = f"⚠️ 分析レポート生成に失敗しました: {type(e).__name__}: {e}"
                            report_container.markdown(full_text)

                    st.session_state.generated_report = full_text
        else:
            # 既存レポートをスクロール可能なコンテナで表示
            with st.container(height=400, border=True):
                st.markdown(st.session_state.generated_report)
            if st.button("🔄 レポート再作成"):
                st.session_state.generated_report = None
                st.rerun()

    # --- B. 自動修復 & チャット ---
    st.markdown("---")
    st.subheader("🤖 Remediation & Chat")

    if selected_incident_candidate and selected_incident_candidate["prob"] > 0.6:
        st.markdown(f"""
        <div style="background-color:#e8f5e9;padding:10px;border-radius:5px;border:1px solid #4caf50;color:#2e7d32;margin-bottom:10px;">
            <strong>✅ AI Analysis Completed</strong><br>
            特定された原因 <b>{selected_incident_candidate['id']}</b> に対する復旧手順が利用可能です。<br>
            (リスクスコア: <span style="font-size:1.2em;font-weight:bold;">{selected_incident_candidate['prob']*100:.0f}</span>)
        </div>
        """, unsafe_allow_html=True)

        if "remediation_plan" not in st.session_state:
            if st.button("✨ 修復プランを作成 (Generate Fix)"):
                 if "generated_report" not in st.session_state or not st.session_state.generated_report:
                     st.warning("先に「📝 詳細レポートを作成 (Generate Report)」を実行してください。")
                 else:
                     # ★改修: Analyst Report の結論から Remediation を生成（プロンプト完全分離）
                     remediation_container = st.empty()
                     
                     t_node = TOPOLOGY.get(selected_incident_candidate["id"])
                     
                     # Remediation用キャッシュキーを生成（Analyst Reportとは別）
                     cache_key_remediation = "|".join([
                         "remediation",  # Remediation専用キー
                         selected_scenario,
                         str(selected_incident_candidate.get("id")),
                         _hash_text(st.session_state.generated_report or ""),
                     ])
                     
                     if cache_key_remediation in st.session_state.report_cache:
                         remediation_text = st.session_state.report_cache[cache_key_remediation]
                         remediation_container.markdown(remediation_text)
                     else:
                         try:
                             # ★パフォーマンス改善: ストリーミング対応 + エラーハンドリング強化
                             remediation_container.write("🤖 復旧プラン生成中...")
                             placeholder = remediation_container.empty()
                             remediation_text = ""
                             error_occurred = False
                             
                             # ストリーミングで段階的に取得・表示
                             try:
                                 for chunk in generate_remediation_commands_streaming(
                                     scenario=selected_scenario,
                                     analysis_result=st.session_state.generated_report or "",
                                     target_node=t_node,
                                     api_key=api_key,
                                     max_retries=2,  # リトライ回数
                                     backoff=3       # リトライ間隔
                                 ):
                                     remediation_text += chunk
                                     placeholder.markdown(remediation_text)  # 段階的に更新
                             except google_exceptions.ServiceUnavailable:
                                 error_occurred = True
                                 remediation_text += "\n\n⚠️ **API が混雑しています。生成済みプランを表示します。**"
                                 placeholder.markdown(remediation_text)
                             
                             if not remediation_text or remediation_text.startswith("Error"):
                                 remediation_text = f"⚠️ 復旧プラン生成に失敗しました: {remediation_text}"
                                 placeholder.markdown(remediation_text)
                             
                             # 部分的でもキャッシュに保存
                             st.session_state.report_cache[cache_key_remediation] = remediation_text
                         except google_exceptions.ServiceUnavailable:
                             remediation_text = "⚠️ 現在、AIモデルが混雑しています (503 Error)。時間を置いて再度お試しください。"
                             remediation_container.markdown(remediation_text)
                         except Exception as e:
                             remediation_text = f"⚠️ 復旧プラン生成に失敗しました: {type(e).__name__}: {e}"
                             remediation_container.markdown(remediation_text)
                     
                     st.session_state.remediation_plan = remediation_text
                     st.rerun()
        
        if "remediation_plan" in st.session_state:
            # Remediation planをスクロール可能なコンテナで表示
            with st.container(height=400, border=True):
                st.info("AI Generated Recovery Procedure（復旧手順）")
                st.markdown(st.session_state.remediation_plan)
            
            col_exec1, col_exec2 = st.columns(2)
            
            with col_exec1:
                if st.button("🚀 修復実行 (Execute)", type="primary"):
                    if not api_key:
                        st.error("API Key Required")
                    else:
                        with st.status("Autonomic Remediation in progress...", expanded=True) as status:
                            target_node_obj = TOPOLOGY.get(selected_incident_candidate["id"])
                            device_info = target_node_obj.metadata if target_node_obj else {}
                            
                            # ★改善案C: 並列実行（実運用対応版）
                            st.write("🔄 Executing remediation steps in parallel...")
                            
                            results = run_remediation_parallel_v2(
                                device_id=selected_incident_candidate["id"],
                                device_info=device_info,
                                scenario=selected_scenario,
                                environment=RemediationEnvironment.DEMO,  # 本番はPROD
                                timeout_per_step=30
                            )
                            
                            # 結果を表示
                            st.write("📋 Remediation steps result:")
                            
                            all_success = True
                            remediation_summary = []
                            
                            for step_name in ["Backup", "Apply", "Verify"]:
                                result = results.get(step_name)
                                if result:
                                    st.write(str(result))
                                    remediation_summary.append(str(result))
                                    if result.status != "success":
                                        all_success = False
                            
                            # 統合ログを作成
                            verification_log = "\n".join(remediation_summary)
                            st.session_state.verification_log = verification_log
                            st.session_state.remediation_results = results
                            
                            if all_success:
                                st.write("✅ All remediation steps completed successfully.")
                                status.update(label="Process Finished", state="complete", expanded=False)
                            else:
                                st.write("⚠️ Some remediation steps failed. Please review.")
                                status.update(label="Process Finished - With Errors", state="error", expanded=True)
                        
                        if all_success:
                            st.success("Remediation Process Finished.")

            with col_exec2:
                 if st.button("キャンセル"):
                    del st.session_state.remediation_plan
                    st.session_state.verification_log = None
                    st.rerun()
            
            if st.session_state.get("verification_log"):
                st.markdown("#### 🔎 Post-Fix Verification Logs")
                st.code(st.session_state.verification_log, language="text")
                
                is_success = "up" in st.session_state.verification_log.lower() or "ok" in st.session_state.verification_log.lower()
                
                if is_success:
                    # 復旧成功フラグ（デモ用）。次回の「診断実行」で成功側の疑似ログを返します。
                    st.session_state.recovered_devices = st.session_state.get("recovered_devices") or {}
                    st.session_state.recovered_scenario_map = st.session_state.get("recovered_scenario_map") or {}
                    st.session_state.recovered_devices[target_device_id] = True
                    st.session_state.recovered_scenario_map[target_device_id] = selected_scenario
                    
                    # バルーンは一度だけ表示
                    if not st.session_state.balloons_shown:
                        st.balloons()
                        st.session_state.balloons_shown = True
                    
                    st.success("✅ System Recovered Successfully!")
                else:
                    st.warning("⚠️ Verification indicates potential issues. Please check manually.")

                if st.button("デモを終了してリセット"):
                    del st.session_state.remediation_plan
                    st.session_state.verification_log = None
                    st.session_state.current_scenario = "正常稼働"
                    st.session_state.balloons_shown = False  # バルーンフラグもリセット
                    st.rerun()
    else:
        if selected_incident_candidate:
            device_id = selected_incident_candidate.get('id', '')
            score = selected_incident_candidate['prob'] * 100
            
            # 正常稼働時（SYSTEM, prob=0）は緑色メッセージ
            if device_id == "SYSTEM" and score == 0:
                st.markdown("""
                <div style="background-color:#e8f5e9;padding:10px;border-radius:5px;border:1px solid #4caf50;color:#2e7d32;margin-bottom:10px;">
                    <strong>✅ 正常稼働中</strong><br>
                    現在、ネットワークは正常に稼働しています。対応が必要なインシデントはありません。
                </div>
                """, unsafe_allow_html=True)
            else:
                # 低リスクスコア時はオレンジ表示
                st.markdown(f"""
                <div style="background-color:#fff3e0;padding:10px;border-radius:5px;border:1px solid #ff9800;color:#e65100;margin-bottom:10px;">
                    <strong>⚠️ 監視中</strong><br>
                    対象: <b>{device_id}</b><br>
                    (リスクスコア: {score:.0f} - 60以上で自動修復を推奨)
                </div>
                """, unsafe_allow_html=True)

    # チャット (常時表示) - オプション3で改善
    with st.expander("💬 Chat with AI Agent", expanded=False):
        # 対象CIのサマリ（表示のみ、UXは崩さず最小）
        _chat_target_id = ""
        try:
            if selected_incident_candidate:
                _chat_target_id = selected_incident_candidate.get("id", "") or ""
        except Exception:
            _chat_target_id = ""
        if not _chat_target_id:
            _chat_target_id = target_device_id if 'target_device_id' in globals() else ""
        _chat_ci = _build_ci_context_for_chat(_chat_target_id) if _chat_target_id else {}
        if _chat_ci:
            _vendor = _chat_ci.get("vendor", "") or "Unknown"
            _os = _chat_ci.get("os", "") or "Unknown"
            _model = _chat_ci.get("model", "") or "Unknown"
            st.caption(f"対象機器: {_chat_target_id}   Vendor: {_vendor}   OS: {_os}   Model: {_model}")

        # クイック質問（入力欄は変えず、コピペ用に提示）
        q1, q2, q3 = st.columns(3)
        if "chat_quick_text" not in st.session_state:
            st.session_state.chat_quick_text = ""

        with q1:
            if st.button("設定バックアップ", use_container_width=True):
                st.session_state.chat_quick_text = "この機器で、現在の設定を安全にバックアップする手順とコマンド例を教えてください。"
        with q2:
            if st.button("ロールバック", use_container_width=True):
                st.session_state.chat_quick_text = "この機器で、変更をロールバックする代表的な手順（候補）と注意点を教えてください。"
        with q3:
            if st.button("確認コマンド", use_container_width=True):
                st.session_state.chat_quick_text = "今回の症状を切り分けるために、まず実行すべき確認コマンド（show/diagnostic）を優先度順に教えてください。"

        if st.session_state.chat_quick_text:
            st.info("クイック質問（コピーして貼り付け）")
            st.code(st.session_state.chat_quick_text)

        if st.session_state.chat_session is None and api_key:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(GEMMA_MODEL)
            st.session_state.chat_session = model.start_chat(history=[])

        # タブでレイアウトを整理
        tab1, tab2 = st.tabs(["💬 会話", "📝 履歴"])
        
        with tab1:
            # 最新の応答のみを目立たせる
            if st.session_state.messages:
                last_msg = st.session_state.messages[-1]
                if last_msg["role"] == "assistant":
                    st.info("🤖 最新の回答")
                    with st.container(height=300):
                        st.markdown(last_msg["content"])
            
            # 入力欄を上部に配置
            st.markdown("---")
            prompt = st.text_area(
                "質問を入力してください:",
                height=70,
                placeholder="Ctrl+Enter または 送信ボタンで送信",
                key="chat_textarea"
            )
            
            col1, col2, col3 = st.columns([3, 1, 1])
            with col2:
                send_button = st.button("送信", type="primary", use_container_width=True)
            with col3:
                if st.button("クリア"):
                    st.session_state.messages = []
                    st.rerun()
            
            # 送信処理
            if send_button and prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                if st.session_state.chat_session:
                    # CI-aware prompt（CI/Config をフル活用）
                    target_id = ""
                    try:
                        if selected_incident_candidate:
                            target_id = selected_incident_candidate.get("id", "") or ""
                    except Exception:
                        target_id = ""
                    if not target_id:
                        try:
                            target_id = target_device_id
                        except Exception:
                            target_id = ""
                    ci = _build_ci_context_for_chat(target_id) if target_id else {}
                    ci_prompt = f"""あなたはネットワーク運用（NOC/SRE）の実務者です。
次の CI 情報と Config 抜粋を必ず参照して、具体的に回答してください。一般論だけで終わらせないでください。

【CI (JSON)】
{json.dumps(ci, ensure_ascii=False, indent=2)}

【ユーザーの質問】
{prompt}

回答ルール:
- CI/Config に基づく具体手順・コマンド例を提示する
- 追加確認が必要なら、質問は最小限（1〜2点）に絞る
- 不明な前提は推測せず「CIに無いので確認が必要」と明記する
"""
                    
                    with st.spinner("AI が回答を生成中..."):
                        try:
                            response = generate_content_with_retry(st.session_state.chat_session.model, ci_prompt, stream=False)
                            if response:
                                full_response = response.text if hasattr(response, "text") else str(response)
                                if not full_response.strip():
                                    full_response = "AI応答が空でした（CIは渡しましたが出力が生成されませんでした）。"
                                st.session_state.messages.append({"role": "assistant", "content": full_response})
                            else:
                                st.error("AIからの応答がありませんでした。")
                        except Exception as e:
                            st.error(f"エラーが発生しました: {e}")
                    st.rerun()
        
        with tab2:
            # スクロール可能な履歴表示
            if st.session_state.messages:
                history_container = st.container(height=400)
                with history_container:
                    for i, msg in enumerate(st.session_state.messages):
                        icon = "🤖" if msg["role"] == "assistant" else "👤"
                        with st.container(border=True):
                            st.markdown(f"{icon} **{msg['role'].upper()}** (メッセージ {i+1})")
                            st.markdown(msg["content"])
            else:
                st.info("会話履歴はまだありません。")

# ベイズ更新トリガー (診断後)
if st.session_state.trigger_analysis and st.session_state.live_result:
    if st.session_state.verification_result:
        pass
    st.session_state.trigger_analysis = False
    st.rerun()
