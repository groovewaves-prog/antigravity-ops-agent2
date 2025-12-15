import streamlit as st
import graphviz
import os
import time
import google.generativeai as genai

from data import TOPOLOGY
from logic import CausalInferenceEngine, Alarm, simulate_cascade_failure
from network_ops import run_diagnostic_simulation, generate_config_from_intent, generate_health_check_commands, generate_remediation_commands
from verifier import verify_log_content, format_verification_report

# ★新規追加モジュールのインポート
from dashboard import render_intelligent_alarm_viewer
from bayes_engine import BayesianRCA

# --- ページ設定 ---
st.set_page_config(page_title="Antigravity Live", page_icon="⚡", layout="wide")

# --- 関数: トポロジー図の生成 (修正版) ---
def render_topology(alarms, root_cause_node, root_severity="CRITICAL"):
    graph = graphviz.Digraph()
    graph.attr(rankdir='TB')
    graph.attr('node', shape='box', style='rounded,filled', fontname='Helvetica')
    
    # アラーム辞書（ID -> Alarmオブジェクト）を作成
    alarm_map = {a.device_id: a for a in alarms}
    alarmed_ids = set(alarm_map.keys())
    
    for node_id, node in TOPOLOGY.items():
        color = "#e8f5e9" # Default Green
        penwidth = "1"
        fontcolor = "black"
        label = f"{node_id}\n({node.type})"
        
        red_type = node.metadata.get("redundancy_type")
        if red_type:
            label += f"\n[{red_type} Redundancy]"
        
        vendor = node.metadata.get("vendor")
        if vendor:
            label += f"\n[{vendor}]"

        # 根本原因ノードの描画
        if root_cause_node and node_id == root_cause_node.id:
            # ロジック判定(root_severity)ではなく、個別のAlarm重要度を優先して色を決める
            this_alarm = alarm_map.get(node_id)
            node_severity = this_alarm.severity if this_alarm else root_severity
            
            if node_severity == "CRITICAL":
                color = "#ffcdd2" # Red (Down)
            elif node_severity == "WARNING":
                color = "#fff9c4" # Yellow (Warning)
            else:
                color = "#e8f5e9"
            
            penwidth = "3"
            label += "\n[ROOT CAUSE]"
            
        elif node_id in alarmed_ids:
            # 連鎖アラーム等は黄色
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

# --- 関数: Config自動読み込み ---
def load_config_by_id(device_id):
    path = f"configs/{device_id}.txt"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None
    return None

# --- UI構築 ---
st.title("⚡ Antigravity AI Agent (Autonomous Demo)")

api_key = None
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = os.environ.get("GOOGLE_API_KEY")

# --- サイドバー ---
with st.sidebar:
    st.header("⚡ 運用モード選択")
    app_mode = st.radio("機能選択:", ("🚨 障害対応", "🔧 設定生成"))
    st.markdown("---")
    
    selected_scenario = "正常稼働"
    
    if app_mode == "🚨 障害対応":
        SCENARIO_MAP = {
            "基本・広域障害": ["正常稼働", "1. WAN全回線断", "2. FW片系障害", "3. L2SWサイレント障害"],
            "WAN Router": ["4. [WAN] 電源障害：片系", "5. [WAN] 電源障害：両系", "6. [WAN] BGPルートフラッピング", "7. [WAN] FAN故障", "8. [WAN] メモリリーク"],
            "Firewall (Juniper)": ["9. [FW] 電源障害：片系", "10. [FW] 電源障害：両系", "11. [FW] FAN故障", "12. [FW] メモリリーク"],
            "L2 Switch": ["13. [L2SW] 電源障害：片系", "14. [L2SW] 電源障害：両系", "15. [L2SW] FAN故障", "16. [L2SW] メモリリーク"],
            "Live": ["99. [Live] Cisco実機診断"]
        }
        selected_category = st.selectbox("対象カテゴリ:", list(SCENARIO_MAP.keys()))
        selected_scenario = st.radio("発生シナリオ:", SCENARIO_MAP[selected_category])
    
    if api_key:
        st.success("API Connected")
    else:
        st.warning("API Key Missing")
        user_key = st.text_input("Google API Key", type="password")
        if user_key: api_key = user_key

# --- セッション管理 ---
if "current_mode" not in st.session_state:
    st.session_state.current_mode = app_mode
    st.session_state.messages = []
    st.session_state.chat_session = None 
    st.session_state.live_result = None
    st.session_state.trigger_analysis = False
    st.session_state.verification_result = None

if st.session_state.current_mode != app_mode:
    st.session_state.current_mode = app_mode
    st.session_state.messages = []
    st.rerun()

# ==========================================
# モードA: 障害対応
# ==========================================
if app_mode == "🚨 障害対応":
    
    if "current_scenario" not in st.session_state:
        st.session_state.current_scenario = "正常稼働"
    
    if st.session_state.current_scenario != selected_scenario:
        st.session_state.current_scenario = selected_scenario
        st.session_state.messages = []
        st.session_state.chat_session = None
        st.session_state.live_result = None
        st.session_state.trigger_analysis = False
        st.session_state.verification_result = None
        # 修復プランもリセット
        if "remediation_plan" in st.session_state: del st.session_state.remediation_plan
        # ベイズエンジンのリセット（新しいシナリオ用に再初期化）
        if "bayes_engine" in st.session_state: del st.session_state.bayes_engine
        st.rerun()

    alarms = []
    root_severity = "CRITICAL"
    target_device_id = None

    # --- アラーム生成ロジック ---
    if "WAN全回線断" in selected_scenario:
        target_device_id = "WAN_ROUTER_01"
        alarms = simulate_cascade_failure("WAN_ROUTER_01", TOPOLOGY)
    elif "FW片系障害" in selected_scenario:
        target_device_id = "FW_01_PRIMARY"
        alarms = [Alarm("FW_01_PRIMARY", "Heartbeat Loss", "WARNING")]
        root_severity = "WARNING"
    elif "L2SWサイレント障害" in selected_scenario:
        target_device_id = "L2_SW_01"
        alarms = [Alarm("AP_01", "Connection Lost", "CRITICAL"), Alarm("AP_02", "Connection Lost", "CRITICAL")]
    else:
        if "[WAN]" in selected_scenario: target_device_id = "WAN_ROUTER_01"
        elif "[FW]" in selected_scenario: target_device_id = "FW_01_PRIMARY"
        elif "[L2SW]" in selected_scenario: target_device_id = "L2_SW_01"

        if target_device_id:
            if "電源障害：片系" in selected_scenario:
                alarms = [Alarm(target_device_id, "Power Supply 1 Failed", "WARNING")]
                root_severity = "WARNING"
            elif "電源障害：両系" in selected_scenario:
                if target_device_id == "FW_01_PRIMARY":
                    alarms = [Alarm(target_device_id, "Power Supply: Dual Loss (Device Down)", "CRITICAL")]
                else:
                    alarms = simulate_cascade_failure(target_device_id, TOPOLOGY, "Power Supply: Dual Loss (Device Down)")
                root_severity = "CRITICAL"
            elif "BGP" in selected_scenario:
                alarms = [Alarm(target_device_id, "BGP Flapping", "WARNING")]
                root_severity = "WARNING"
            elif "FAN" in selected_scenario:
                alarms = [Alarm(target_device_id, "Fan Fail", "WARNING")]
                root_severity = "WARNING"
            elif "メモリ" in selected_scenario:
                alarms = [Alarm(target_device_id, "Memory High", "WARNING")]
                root_severity = "WARNING"

    # --- ベイズエンジン初期化 ---
    if "bayes_engine" not in st.session_state:
        st.session_state.bayes_engine = BayesianRCA(TOPOLOGY)
        # 初期アラームを証拠として投入
        if "BGP" in selected_scenario:
             st.session_state.bayes_engine.update_probabilities("alarm", "BGP Flapping")
        elif "全回線断" in selected_scenario:
             st.session_state.bayes_engine.update_probabilities("ping", "NG")

    # --- 診断と分析の実行 ---
    is_live_mode = ("[Live]" in selected_scenario)
    
    if st.button("🚀 診断実行 (Auto-Diagnostic)", type="primary"):
        if not api_key:
            st.error("API Key Required")
        else:
            with st.status("Agent Operating...", expanded=True) as status:
                st.write("🔌 Executing Diagnostics...")
                target_node_obj = TOPOLOGY.get(target_device_id) if target_device_id else None
                res = run_diagnostic_simulation(selected_scenario, target_node_obj, api_key)
                
                st.session_state.live_result = res
                if res["status"] == "SUCCESS":
                    st.write("✅ Data Acquired.")
                    status.update(label="Complete!", state="complete", expanded=False)
                    
                    log_content = res.get('sanitized_log', "")
                    verification = verify_log_content(log_content)
                    st.session_state.verification_result = verification
                    
                elif res["status"] == "SKIPPED":
                    status.update(label="Skipped", state="complete")
                else:
                    st.write("❌ Check Failed.")
                    status.update(label="Target Unreachable", state="error", expanded=False)
                    st.session_state.verification_result = {
                        "ping_status": "N/A (Connection Failed)",
                        "interface_status": "Unknown",
                        "hardware_status": "Unknown",
                        "error_keywords": "Connection Error"
                    }
                
                st.session_state.trigger_analysis = True
    
    # --- 分析完了後のベイズ更新 ---
    if st.session_state.trigger_analysis and st.session_state.live_result:
        res = st.session_state.live_result
        if res["status"] == "SUCCESS":
            if st.session_state.verification_result:
                v_res = st.session_state.verification_result
                
                # 証拠投入: Ping結果
                if "NG" in v_res.get("ping_status", ""):
                        st.session_state.bayes_engine.update_probabilities("ping", "NG")
                
                # 証拠投入: ログキーワード
                if "DOWN" in v_res.get("interface_status", ""):
                        st.session_state.bayes_engine.update_probabilities("log", "Interface Down")
        
        st.session_state.trigger_analysis = False # フラグリセット
        st.rerun() # 画面更新

    # --- メインダッシュボード描画 ---
    
    # 1. 新しいインシデントビューアー
    top_cause_candidate = None
    if "bayes_engine" in st.session_state:
        top_cause_candidate = render_intelligent_alarm_viewer(st.session_state.bayes_engine, selected_scenario)

    col_map, col_action = st.columns([3, 2])

    with col_map:
        st.subheader("🌐 Impact Topology")
        
        # 従来のCausalInferenceEngineも一応動かす（フォールバック用）
        rule_based_root = None
        rule_based_severity = root_severity
        
        # ビューアーでトップの原因が特定されていれば、それを優先してマップを描画
        current_root_node = None
        current_severity = "WARNING"
        
        if top_cause_candidate and top_cause_candidate["prob"] > 0.6:
            current_root_node = TOPOLOGY.get(top_cause_candidate["id"])
            current_severity = "CRITICAL"
        elif target_device_id:
             current_root_node = TOPOLOGY.get(target_device_id)
             current_severity = root_severity

        st.graphviz_chart(render_topology(alarms, current_root_node, current_severity), use_container_width=True)
        
        if st.session_state.live_result and st.session_state.live_result["status"] == "SUCCESS":
             with st.expander("📄 取得ログ (Sanitized)", expanded=False):
                st.code(st.session_state.live_result["sanitized_log"], language="text")

    with col_action:
        st.subheader("🤖 Closed Loop Automation")
        
        if top_cause_candidate and top_cause_candidate["prob"] > 0.8:
            # 確信度が十分高い場合のみ、修復アクションを有効化
            
            st.success(f"AI has identified the Root Cause: **{top_cause_candidate['id']}**")
            st.info(f"Reason: High correlation with '{top_cause_candidate['type']}' patterns.")
            
            # --- ここからが「自律修復」 ---
            if "remediation_plan" not in st.session_state:
                if st.button("✨ Generate Remediation Plan (修復案作成)", type="primary"):
                    if not api_key:
                        st.error("API Key Required")
                    else:
                        with st.spinner("AI is generating recovery commands..."):
                            target_node = TOPOLOGY.get(top_cause_candidate["id"])
                            cmds = generate_remediation_commands(
                                selected_scenario, 
                                f"Root cause identified as {top_cause_candidate['type']}", 
                                target_node, 
                                api_key
                            )
                            st.session_state.remediation_plan = cmds
                            st.rerun()
            
            if "remediation_plan" in st.session_state:
                st.markdown("##### 🛠️ AI Proposed Actions")
                st.code(st.session_state.remediation_plan, language="cisco")
                
                col_exec1, col_exec2 = st.columns(2)
                with col_exec1:
                    if st.button("🚀 Execute Fix (修復実行)", type="primary"):
                        with st.status("Autonomic Remediation in progress...", expanded=True) as status:
                            st.write("📡 Connecting to device via Netmiko...")
                            time.sleep(1)
                            st.write("⚙️ Applying configuration...")
                            time.sleep(1)
                            st.write("✅ Verifying service recovery...")
                            time.sleep(1)
                            status.update(label="System Restored Successfully!", state="complete", expanded=False)
                        
                        st.balloons()
                        st.success("障害は解消されました。クローズドループ完了。")
                        
                        if st.button("Reset Demo"):
                            del st.session_state.remediation_plan
                            st.session_state.current_scenario = "正常稼働"
                            st.rerun()
                
                with col_exec2:
                    if st.button("❌ Reject"):
                        del st.session_state.remediation_plan
                        st.rerun()
        
        else:
            st.caption("Waiting for higher confidence score to enable automation...")

# ... (モードB: 設定生成は変更なし) ...
elif app_mode == "🔧 設定生成":
    st.subheader("🔧 Intent-Based Config Generator")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.info("自然言語の指示(Intent)から、メーカー仕様に合わせたConfigを自動生成します。")
        target_id = st.selectbox("対象機器を選択:", list(TOPOLOGY.keys()))
        target_node = TOPOLOGY[target_id]
        vendor = target_node.metadata.get("vendor", "Unknown")
        st.caption(f"Device Info: {vendor}")
        current_conf = load_config_by_id(target_id)
        with st.expander("現在のConfigを確認"):
            st.code(current_conf if current_conf else "(No current config)")
        intent = st.text_area("Intent:", height=150, placeholder="例: Gi0/1にVLAN100を割り当てて。")
        if st.button("✨ Config生成", type="primary"):
            if not api_key or not intent:
                st.error("API Key or Intent Missing")
            else:
                with st.spinner("Generating..."):
                    generated_conf = generate_config_from_intent(target_node, current_conf, intent, api_key)
                    st.session_state.generated_conf = generated_conf
    with c2:
        st.subheader("📝 Generated Config")
        if "generated_conf" in st.session_state:
            st.markdown(st.session_state.generated_conf)
            st.success("生成完了")
        else:
            st.info("左側のフォームから指示を入力してください。")
        st.markdown("---")
        st.subheader("🔍 Health Check Commands")
        if st.button("正常性確認コマンドを生成"):
             if not api_key:
                 st.error("API Key Required")
             else:
                 with st.spinner("Generating..."):
                     cmds = generate_health_check_commands(target_node, api_key)
                     st.code(cmds, language="text")
