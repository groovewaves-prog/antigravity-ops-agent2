import streamlit as st
import graphviz
import os
import time
import random
import google.generativeai as genai
import json
import hashlib
from dataclasses import asdict
import pandas as pd
from google.api_core import exceptions as google_exceptions

# モジュール群のインポート
from data import TOPOLOGY
from logic import CausalInferenceEngine, Alarm, simulate_cascade_failure
from network_ops import run_diagnostic_simulation, generate_remediation_commands, predict_initial_symptoms, generate_fake_log_by_ai
from verifier import verify_log_content, format_verification_report
from inference_engine import LogicalRCA

# --- ページ設定 ---


# -----------------------------
# Helpers (keep IF logic minimal)
# -----------------------------
import hashlib

def _hash_text(s: str) -> str:
    """Stable short hash for caching keys."""
    if s is None:
        s = ""
    if not isinstance(s, str):
        s = str(s)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

st.set_page_config(page_title="Antigravity Autonomous", page_icon="⚡", layout="wide")

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

def generate_content_with_retry(model, prompt, stream=True, retries=5, base_sleep=1.5, max_sleep=12.0):
    """Gemini呼び出しのリトライ（503/429/一時障害）とエラーメッセージ整形。
    - ServiceUnavailable(503), ResourceExhausted(429) を主対象に指数バックオフで再試行
    - stream=True の場合も同様に generate_content を呼び出す（返り値は iterator ）
    """
    last_exc = None
    for i in range(retries):
        try:
            return model.generate_content(prompt, stream=stream)
        except (
            google_exceptions.ServiceUnavailable,
            google_exceptions.ResourceExhausted,
            google_exceptions.DeadlineExceeded,
            google_exceptions.InternalServerError,
        ) as e:
            # 典型的には「混雑」「一時的な内部障害」「レート制限」
            last_exc = e
            if i == retries - 1:
                raise
            # exponential backoff + small jitter
            sleep_s = min(max_sleep, base_sleep * (2 ** i)) * (0.85 + random.random() * 0.3)
            time.sleep(sleep_s)
        except (
            google_exceptions.Unauthenticated,
            google_exceptions.PermissionDenied,
            google_exceptions.InvalidArgument,
        ) as e:
            # APIキー/権限/リクエスト不正はリトライしても改善しにくい
            raise
        except Exception as e:
            last_exc = e
            if i == retries - 1:
                raise
            time.sleep(min(max_sleep, base_sleep * (2 ** i)))
    raise last_exc if last_exc else RuntimeError("Unknown generation error")



# ==========================================
# LLM最小化: 生成物バンドル/キャッシュ/パース
# ==========================================
def _stable_hash(s: str) -> str:
    try:
        return hashlib.sha1((s or "").encode("utf-8")).hexdigest()
    except Exception:
        return "0"*40

def _extract_section_by_h3(md_text: str, heading_keyword: str) -> str:
    """### 見出し単位でセクション本文を抜き出す（次の###まで）。見つからなければ空文字。"""
    if not md_text:
        return ""
    lines = md_text.splitlines()
    out = []
    in_sec = False
    for line in lines:
        if line.strip().startswith("### "):
            if in_sec:
                break
            if heading_keyword in line:
                in_sec = True
                out.append(line)
                continue
        if in_sec:
            out.append(line)
    return "\n".join(out).strip()

def _extract_expectations(md_text: str) -> str:
    # 期待結果/合否判定系を優先的に拾う
    for key in ["期待結果", "合否判定", "Acceptance", "PASS/FAIL"]:
        sec = _extract_section_by_h3(md_text, key)
        if sec:
            return sec
    # fallback: 見出しが無い場合は空
    return ""

def _generate_bundle_prompt(selected_scenario: str, cand: dict, topology_context: dict, target_conf: str, verification_context: str, force_polite_style: bool = False) -> str:
    return f"""あなたは熟練したネットワーク運用エンジニアです。
以下の障害インシデントについて、**運用者向けの成果物を1つの回答にまとめて**作成してください。

【重要方針】
- 出力は必ず「です／ます調」で統一してください。
- 顧客向けの定型句は禁止です（例:「原因究明と復旧作業を最優先で進めております」「随時ご報告いたします」など）。
- 憶測で断定しません。推定する場合は根拠（観測事実）を併記してください。
- コマンドは必ずMarkdownのコードブロックで囲ってください。
- **期待結果（合否判定キー）**を必ず含めてください（例: "show interfaceでup/upが確認できる"、"BGPがEstablished"、"pingが0% loss" など）。
- すべての成果物を1回の応答にまとめること（追加の質問・追加の出力はしない）。

【入力情報】
- 発生シナリオ: {selected_scenario}
- 根本原因候補: {cand.get('id')} ({cand.get('label')})
- リスクスコア: {cand.get('prob',0)*100:.0f}

- CI/トポロジー情報(JSON):
{json.dumps(topology_context, ensure_ascii=False, indent=2)}

- 能動的診断結果（あれば）:
{verification_context or "特になし"}

- 対象機器Config（抜粋・あれば）:
{(target_conf or "特になし")[:2000]}

【出力フォーマット（厳守）】
### 運用状況報告
- 1. 観測事実
- 2. 影響範囲（トポロジーから）
- 3. 暫定原因と根拠
- 4. 次の確認コマンド（期待結果つき）
- 5. 切り分け手順（判断条件つき）

### 復旧手順書
#### 1. 物理対応（必要な場合）
#### 2. 復旧コマンド (Recovery Config)
```bash
# commands...
```
#### 3. 正常性確認コマンド (Verification Commands)
```bash
# commands...
```
#### 4. 期待結果（合否判定キー）
- コマンドごとに、PASS条件/FAIL条件を箇条書きで明示してください。

"""


def _simulate_verification_log(device_id: str, scenario: str) -> str:
    # LLMを使わない簡易な疑似ログ（verifier.py のヒューリスティックが拾える語を含める）
    base = [
        f"DEVICE={device_id}",
        f"SCENARIO={scenario}",
        "PING: OK (0% loss)",
        "INTERFACE: UP/UP",
        "BGP: Established",
        "HEALTHCHECK: OK",
    ]
    return "\n".join(base)

def _ensure_cmd_state():
    if "recovery_commands" not in st.session_state:
        st.session_state.recovery_commands = ""
    if "verification_commands" not in st.session_state:
        st.session_state.verification_commands = ""
    if "active_probe_logs" not in st.session_state:
        st.session_state.active_probe_logs = {}  # device_id -> log(text)

def _extract_first_codeblock_after_heading(markdown_text: str, heading_keyword: str) -> str:
    """見出し（例: '復旧コマンド'）以降で最初に出現するコードブロックを抽出。"""
    if not markdown_text:
        return ""
    # heading_keyword を含む行を探す（### ...）
    lines = markdown_text.splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        if heading_keyword in line:
            start_idx = i
            break
    # その後の ``` を探す
    in_block = False
    block_lines = []
    for line in lines[start_idx:]:
        if line.strip().startswith("```") and not in_block:
            in_block = True
            continue
        if line.strip().startswith("```") and in_block:
            break
        if in_block:
            block_lines.append(line)
    return "\n".join(block_lines).strip()

def _friendly_ai_error_message(e: Exception) -> str:
    # 503/429 と APIキー系を切り分けて運用者に分かる形にする
    msg = str(e)
    cls = e.__class__.__name__
    if isinstance(e, google_exceptions.ResourceExhausted) or "429" in msg:
        return "AI API がレート制限（429）に達しました。短時間に連続実行していないか、同一APIキーの同時実行が多くないかを確認してください。"
    if isinstance(e, google_exceptions.ServiceUnavailable) or "503" in msg:
        return "AI API が一時的に 503（Service Unavailable）を返しています。サービス側の混雑/一時障害の可能性が高いです。少し間隔を空けて再試行してください。"
    if isinstance(e, google_exceptions.Unauthenticated) or "401" in msg:
        return "AI API の認証に失敗しました（401）。APIキーが未設定/誤りの可能性があります。"
    if isinstance(e, google_exceptions.PermissionDenied) or "403" in msg:
        return "AI API の権限エラー（403）です。APIキーの権限・プロジェクト設定・利用可能なモデルを確認してください。"
    return f"AI API エラー: {cls}: {msg}"

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
    if api_key: st.success("API Connected")
    else:
        st.warning("API Key Missing")
        user_key = st.text_input("Google API Key", type="password")
        if user_key: api_key = user_key

# --- セッション管理 ---
if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = "正常稼働"

# 変数初期化
_default_session_state = {
    "live_result": None,
    "messages": [],
    "chat_session": None,
    "trigger_analysis": False,
    "verification_result": None,
    "generated_report": None,
    "verification_log": None,
    "last_report_cand_id": None,
    "logic_engine": None,
    "recovery_commands": "",
    "verification_commands": "",
    "active_probe_logs": {},  # device_id -> log(text)
}
for k, v in _default_session_state.items():
    if k not in st.session_state:
        st.session_state[k] = v

_ensure_cmd_state()

# エンジン初期化
if not st.session_state.logic_engine:
    st.session_state.logic_engine = LogicalRCA(TOPOLOGY)

# シナリオ切り替え時のリセット
if st.session_state.current_scenario != selected_scenario:
    st.session_state.current_scenario = selected_scenario
    st.session_state.messages = []      
    st.session_state.chat_session = None 
    st.session_state.live_result = None 
    st.session_state.trigger_analysis = False
    st.session_state.verification_result = None
    st.session_state.generated_report = None
    st.session_state.verification_log = None 
    st.session_state.last_report_cand_id = None
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
st.info("💡 ヒント: インシデントの行をクリックすると、右側に詳細分析と復旧プランが表示されます。")

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
        # NOTE: 疑似プローブ（LLMは呼びません）
        # 診断対象は「現在選択中のインシデント（右の行選択）」を優先
        diag_device_id = None
        if selected_incident_candidate:
            diag_device_id = selected_incident_candidate.get("id")
        if not diag_device_id:
            diag_device_id = target_device_id

        target_node_obj = TOPOLOGY.get(diag_device_id) if diag_device_id else None

        # トポロジーコンテキスト（親子）
        parent_id = getattr(target_node_obj, "parent_id", None) if target_node_obj else None
        children_ids = [
            nid for nid, n in TOPOLOGY.items()
            if getattr(n, "parent_id", None) == diag_device_id
        ] if diag_device_id else []

        # Config 取得（あれば）
        target_conf = ""
        if diag_device_id:
            try:
                target_conf = load_config_by_id(diag_device_id) or ""
            except Exception:
                target_conf = ""

        # Alarm 取得（あれば）
        device_alarms = []
        try:
            device_alarms = [a for a in alarms if a.get("device_id") == diag_device_id]
        except Exception:
            device_alarms = []

        with st.status("Agent Operating...", expanded=True) as status:
            st.write(f"🔌 Connecting to device... [{diag_device_id}]")
            st.write("🧪 Active probe (simulated): collecting alarms/config/topology context")
            # 疑似ログ（サニタイズ済み前提）
            log_lines = []
            log_lines.append(f"DEVICE={diag_device_id}")
            log_lines.append(f"PARENT={parent_id}")
            log_lines.append(f"CHILDREN={children_ids}")
            if device_alarms:
                log_lines.append("ALARMS:")
                for a in device_alarms[:20]:
                    log_lines.append(f"- {a.get('severity','')}: {a.get('message','')}")
            else:
                log_lines.append("ALARMS: (none found for this device in current dataset)")
            if target_conf:
                log_lines.append("CONFIG_SNIPPET:")
                log_lines.append(target_conf[:1500])
            else:
                log_lines.append("CONFIG_SNIPPET: (not available)")

            log_content = "\n".join(probe_lines)
# patched
# 
    log_content = "\n".join(log_lines)
            # 後段（レポート/修復プラン）に渡すために保持
            if diag_device_id:
                st.session_state.active_probe_logs[diag_device_id] = log_content

            verification = verify_log_content(log_content)
            st.session_state.verification_result = verification

            st.write("✅ Log Acquired (simulated) & Stored.")
            status.update(label="Diagnostics Complete!", state="complete", expanded=False)

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
            if st.button("✨ 修復プランを作成 (Generate Fix)", disabled=not bool(st.session_state.get("generated_report"))):
                _ensure_cmd_state()
                cand = selected_incident_candidate
                bundle = st.session_state.get('last_bundle')
                if not bundle or bundle.get('cand_id') != cand.get('id'):
                    st.warning('先に「詳細レポートを作成 (Generate Report)」を実行してください。')
                else:
                    st.session_state.remediation_plan = bundle.get('plan_md') or bundle.get('bundle_md')
                    st.session_state.recovery_commands = bundle.get('recovery_cmds') or st.session_state.recovery_commands
                    st.session_state.verification_commands = bundle.get('verify_cmds') or st.session_state.verification_commands
                    st.session_state.expected_results = bundle.get('expectations')
                    st.success('修復プランを準備しました。')
            with st.container(border=True):
                st.info("AI Generated Recovery Procedure")
                st.markdown(st.session_state.remediation_plan)
            
            col_rec, col_exec1, col_exec2 = st.columns(3)
            
            with col_rec:
                if st.button("🛠️ 復旧コマンド", help="直近で生成された復旧コマンドを表示します"):
                    _ensure_cmd_state()
                    if st.session_state.get("recovery_commands"):
                        st.markdown("#### 🛠️ Recovery Commands")
                        st.code(st.session_state.recovery_commands, language="bash")
                    else:
                        st.warning("復旧コマンドが未生成です。先に Generate Fix または Generate Report を実行してください。")

            with col_exec1:
                if st.button("🚀 修復実行 (Execute)", type="primary"):
                    if not api_key:
                        st.error("API Key Required")
                    else:
                        with st.status("Autonomic Remediation in progress...", expanded=True) as status:
                            st.write("⚙️ Applying Recovery Commands (simulated)...")
                            _ensure_cmd_state()
                            if st.session_state.get("recovery_commands"):
                                st.code(st.session_state.recovery_commands, language="bash")
                            else:
                                st.info("復旧コマンドは未生成のため、適用フェーズはスキップします。")

                            time.sleep(1.0)

                            st.write("🔎 Running Verification Commands (simulated)...")
                            if st.session_state.get("verification_commands"):
                                st.code(st.session_state.verification_commands, language="bash")

                            target_node_obj = TOPOLOGY.get(selected_incident_candidate["id"])
                            device_id = (target_node_obj.id if target_node_obj else selected_incident_candidate['id'])
                            verification_log = _simulate_verification_log(device_id, selected_scenario)
                            st.session_state.verification_log = verification_log
                            try:
                                st.session_state.verification_result = verify_log_content(verification_log)
                            except Exception:
                                st.session_state.verification_result = None
                            st.session_state.verification_log = verification_log

                            st.write("✅ Verification Completed.")
                            status.update(label="Process Finished", state="complete", expanded=False)
                        
                        st.success("Remediation Process Finished.")

            



            with col_exec_cmd:
                show_disabled = not bool(st.session_state.get("recovery_commands"))
                if st.button("📎 復旧コマンド", disabled=show_disabled):
                    st.markdown("#### 🧩 Recovery Config（いつでも実行用）")
                    st.code(st.session_state.get("recovery_commands", ""), language="bash")
                    if st.session_state.get("verification_commands"):
                        st.markdown("#### ✅ 正常性確認コマンド（参考）")
                        st.code(st.session_state.get("verification_commands", ""), language="bash")

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
                    st.balloons()
                    st.success("✅ System Recovered Successfully!")
                else:
                    st.warning("⚠️ Verification indicates potential issues. Please check manually.")

                if st.button("デモを終了してリセット"):
                    del st.session_state.remediation_plan
                    st.session_state.verification_log = None
                    st.session_state.current_scenario = "正常稼働"
                    st.rerun()
    else:
        if selected_incident_candidate:
            score = selected_incident_candidate['prob'] * 100
            st.warning(f"""
            ⚠️ **自動修復はロックされています**
            現在選択されているインシデントのリスクスコアは **{score:.0f}** です。
            誤操作防止のため、スコアが 60 以上の時のみ自動修復ボタンが有効化されます。
            """)

    # チャット (常時表示)
    with st.expander("💬 Chat with AI Agent", expanded=False):
        if st.session_state.chat_session is None and api_key and selected_scenario != "正常稼働":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemma-3-12b-it")
            st.session_state.chat_session = model.start_chat(history=[])

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if prompt := st.chat_input("Ask details..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            if st.session_state.chat_session:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        res_container = st.empty()
                        response = generate_content_with_retry(st.session_state.chat_session.model, prompt, stream=True)
                        if response:
                            full_response = ""
                            for chunk in response:
                                full_response += chunk.text
                                res_container.markdown(full_response)
                            st.session_state.messages.append({"role": "assistant", "content": full_response})
                        else:
                            st.error("AIからの応答がありませんでした。")

# ベイズ更新トリガー (診断後)
if st.session_state.trigger_analysis and st.session_state.live_result:
    if st.session_state.verification_result:
        pass
    st.session_state.trigger_analysis = False
    st.rerun()


def _extract_first_codeblock_after_heading(md_text: str, heading_keyword: str):
    """Markdown本文から、指定キーワードを含む見出し以降の最初のコードブロックを返す。見つからなければ None。"""
    if not md_text:
        return None
    lines = md_text.splitlines()
    in_target_section = False
    in_code = False
    buf = []
    for line in lines:
        if line.strip().startswith("#"):
            in_target_section = (heading_keyword in line)
            in_code = False
            buf = []
            continue
        if not in_target_section:
            continue
        if line.strip().startswith("```") and not in_code:
            in_code = True
            buf = []
            continue
        if line.strip().startswith("```") and in_code:
            return "\n".join(buf).strip()
        if in_code:
            buf.append(line)
    return None

def _ensure_cmd_state():
    if "recovery_commands" not in st.session_state:
        st.session_state.recovery_commands = None
    if "verification_commands" not in st.session_state:
        st.session_state.verification_commands = None
