# -*- coding: utf-8 -*-
"""
Google Antigravity AIOps Agent - Streamlit Main Application
完全版: アラーム選別、真因特定、カスケード障害分析
"""

import streamlit as st
import os
import json
import time
from typing import List, Dict, Any
import google.generativeai as genai

# 既存モジュールのインポート
from data import TOPOLOGY, NetworkNode
from logic import CausalInferenceEngine, Alarm, simulate_cascade_failure
from inference_engine import LogicalRCA
from verifier import verify_log_content, format_verification_report
from network_ops import (
    generate_fake_log_by_ai,
    run_diagnostic_simulation,
    generate_remediation_commands,
    generate_health_check_commands
)

# =====================================================
# ページ設定
# =====================================================
st.set_page_config(
    page_title="AIOps - 障害分析システム",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 定数定義
# =====================================================
SCENARIOS = {
    "正常稼働": "正常稼働",
    "--- WANルーター障害 ---": None,
    "[WAN]電源障害：片系": "[WAN]電源障害：片系",
    "[WAN]電源障害：両系": "[WAN]電源障害：両系",
    "[WAN]BGPフラッピング": "[WAN]BGPフラッピング",
    "[WAN]メモリリーク": "[WAN]メモリリーク",
    "--- ファイアウォール ---": None,
    "[FW]電源障害：片系": "[FW]電源障害：片系",
    "[FW]電源障害：両系": "[FW]電源障害：両系",
    "[FW]FAN故障": "[FW]FAN故障",
    "[FW]メモリリーク": "[FW]メモリリーク",
    "--- L2スイッチ ---": None,
    "[L2SW]電源障害：片系": "[L2SW]電源障害：片系",
    "[L2SW]電源障害：両系": "[L2SW]電源障害：両系",
    "[L2SW]FAN故障": "[L2SW]FAN故障",
    "[L2SW]メモリリーク": "[L2SW]メモリリーク",
    "[L2SW]サイレント障害": "[L2SW]サイレント障害",
    "--- アクセスポイント ---": None,
    "[AP]AP_01ダウン": "[AP]AP_01ダウン",
    "[AP]AP_01ケーブル障害": "[AP]AP_01ケーブル障害",
    "--- 多重障害 ---": None,
    "[複合]FW_01_PRIMARYとAP_03の多重障害": "[複合]FW_01_PRIMARYとAP_03の多重障害",
    "[複合]WAN電源片系+FAN多重障害": "[複合]WAN電源片系+FAN多重障害",
}

# =====================================================
# セッション状態の初期化
# =====================================================
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'current_scenario' not in st.session_state:
    st.session_state.current_scenario = None
if 'root_cause_result' not in st.session_state:
    st.session_state.root_cause_result = None
if 'generated_log' not in st.session_state:
    st.session_state.generated_log = ""
if 'remediation_executed' not in st.session_state:
    st.session_state.remediation_executed = False
if 'health_check_done' not in st.session_state:
    st.session_state.health_check_done = False

# =====================================================
# ヘルパー関数
# =====================================================

def get_target_node_from_scenario(scenario: str) -> str:
    """シナリオから対象ノードIDを推定"""
    if "[WAN]" in scenario:
        return "WAN_ROUTER_01"
    elif "[FW]" in scenario:
        return "FW_01_PRIMARY"
    elif "[L2SW]" in scenario:
        return "L2_SW_01"
    elif "[AP]" in scenario:
        return "AP_01"
    elif "FW_01_PRIMARYとAP_03" in scenario:
        return "FW_01_PRIMARY"
    elif "WAN電源" in scenario:
        return "WAN_ROUTER_01"
    return "WAN_ROUTER_01"

def generate_massive_alarms(scenario: str, root_device_id: str) -> List[Alarm]:
    """
    大量の冗長アラームを生成（50-200件）
    実際の運用では、配下の全機器から様々なアラームが上がってくる
    """
    import random
    
    alarms = []
    root_node = TOPOLOGY.get(root_device_id)
    
    if not root_node:
        return alarms
    
    # 根本原因のアラーム
    if "電源" in scenario:
        if "両系" in scenario:
            alarms.append(Alarm(root_device_id, "Power Supply 1 Failed", "CRITICAL"))
            alarms.append(Alarm(root_device_id, "Power Supply 2 Failed", "CRITICAL"))
            alarms.append(Alarm(root_device_id, "Device Unreachable", "CRITICAL"))
        else:
            alarms.append(Alarm(root_device_id, "Power Supply 1 Failed", "WARNING"))
            alarms.append(Alarm(root_device_id, "Redundancy Lost", "WARNING"))
    elif "BGP" in scenario:
        alarms.append(Alarm(root_device_id, "BGP Peer Flapping", "CRITICAL"))
        alarms.append(Alarm(root_device_id, "Route Instability Detected", "WARNING"))
    elif "FAN" in scenario:
        alarms.append(Alarm(root_device_id, "Fan Module Failed", "CRITICAL"))
        alarms.append(Alarm(root_device_id, "Temperature Warning", "WARNING"))
    elif "メモリリーク" in scenario:
        alarms.append(Alarm(root_device_id, "Memory Usage 95%", "CRITICAL"))
        alarms.append(Alarm(root_device_id, "System Performance Degraded", "WARNING"))
    elif "ケーブル" in scenario:
        alarms.append(Alarm(root_device_id, "Interface GigabitEthernet0/1 Down", "CRITICAL"))
        alarms.append(Alarm(root_device_id, "Link Status Changed", "WARNING"))
    elif "ダウン" in scenario:
        alarms.append(Alarm(root_device_id, "Device Down", "CRITICAL"))
        alarms.append(Alarm(root_device_id, "SNMP Timeout", "CRITICAL"))
    
    # カスケード障害のアラーム生成
    cascade_alarms = simulate_cascade_failure(root_device_id, TOPOLOGY, "Connection Lost")
    alarms.extend(cascade_alarms[1:])  # 重複を避けるため根本原因以外を追加
    
    # ノイズアラームを大量追加（50-200件に）
    noise_messages = [
        "SNMP Trap Received",
        "Interface Utilization 50%",
        "Minor Configuration Change",
        "Backup Job Started",
        "User Login Detected",
        "Temperature Normal",
        "Fan Speed Adjusted",
        "ARP Cache Updated",
        "Routing Table Updated",
        "VLAN Database Modified",
        "ACL Hit Count Threshold",
        "Port Security Violation (Info)",
        "NTP Sync OK",
        "DNS Query Timeout (Retry OK)",
        "DHCP Lease Expired (Auto Renewed)",
    ]
    
    target_count = random.randint(50, 200)
    while len(alarms) < target_count:
        random_device = random.choice(list(TOPOLOGY.keys()))
        random_message = random.choice(noise_messages)
        random_severity = random.choice(["INFO", "WARNING", "INFO", "INFO"])  # INFO多め
        alarms.append(Alarm(random_device, random_message, random_severity))
    
    return alarms

def filter_critical_alarms(all_alarms: List[Alarm], api_key: str) -> List[Alarm]:
    """
    AIを使って本当に重要なアラームだけを3-5件に絞る
    """
    if not api_key:
        # APIキーがない場合はCRITICALのみ返す
        return [a for a in all_alarms if a.severity == "CRITICAL"][:5]
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # アラーム情報を整形
    alarm_list = "\n".join([
        f"{i+1}. Device: {a.device_id}, Message: {a.message}, Severity: {a.severity}"
        for i, a in enumerate(all_alarms[:100])  # 最初の100件のみ送信
    ])
    
    prompt = f"""
あなたはネットワーク監視システムのアラームフィルタリングAIです。
以下の大量のアラームから、**根本原因に関連する重要なアラームだけを3〜5件選択**してください。

【アラームリスト】
{alarm_list}

【選択ルール】
1. CRITICAL / WARNING の重要なアラームを優先
2. INFO（情報通知）は基本的に無視
3. 同じデバイスからの重複アラームは1つにまとめる
4. カスケード障害（配下の機器のConnection Lost）は根本原因ではないため除外
5. 電源障害、Interface Down、BGP Flapping、Fan Failなど「直接的な障害」を選ぶ

【出力形式】
選択したアラームの番号をカンマ区切りで出力してください。
例: 1,3,5,12,18

番号のみを出力し、説明は不要です。
"""
    
    try:
        response = model.generate_content(prompt)
        selected_indices = [int(x.strip()) - 1 for x in response.text.strip().split(',')]
        return [all_alarms[i] for i in selected_indices if i < len(all_alarms)]
    except Exception as e:
        st.warning(f"AIフィルタリングエラー: {e}")
        return [a for a in all_alarms if a.severity in ["CRITICAL", "WARNING"]][:5]

def get_cascade_impact(root_device_id: str) -> Dict[str, Any]:
    """
    カスケード障害の影響範囲を分析
    """
    affected_nodes = []
    root_node = TOPOLOGY.get(root_device_id)
    
    if not root_node:
        return {"count": 0, "nodes": [], "reason": ""}
    
    # BFSで配下のノードを列挙
    queue = [root_device_id]
    processed = {root_device_id}
    
    while queue:
        current_id = queue.pop(0)
        children = [n for n in TOPOLOGY.values() if n.parent_id == current_id]
        
        for child in children:
            if child.id not in processed:
                affected_nodes.append(child)
                queue.append(child.id)
                processed.add(child.id)
    
    # 理由文を生成
    reason = f"""
**カスケード障害の詳細分析**

【直接原因】
{root_device_id} が完全にダウンしています。

【なぜ配下の機器が監視不能なのか】
{root_device_id} はネットワークトポロジーのLayer {root_node.layer}に位置し、
すべての通信の中継点となっています。このデバイスがダウンすると、
配下の全機器への通信経路が遮断されるため、監視システムから到達不能となります。

【影響を受けている機器（{len(affected_nodes)}台）】
"""
    
    for node in sorted(affected_nodes, key=lambda n: n.layer):
        reason += f"\n├ {node.id} (Layer {node.layer}, {node.type})"
    
    reason += """

⚠️ **重要な注意事項**
これらの配下の機器自体には障害は発生していません。
ネットワーク経路が遮断されているため「監視不能」状態になっているだけです。
{root_device_id} を復旧すれば、これらの機器は自動的に正常状態に戻ります。
"""
    
    return {
        "count": len(affected_nodes),
        "nodes": affected_nodes,
        "reason": reason
    }

def generate_topology_graph(root_cause_id: str = None, cascade_nodes: List[str] = None) -> str:
    """
    Graphvizフォーマットのトポロジー図を生成
    色分け: 赤=真因、オレンジ=カスケード影響、緑=正常
    """
    cascade_set = set(cascade_nodes) if cascade_nodes else set()
    
    dot = """
digraph Topology {
    rankdir=TB;
    node [shape=box, style=filled];
    
"""
    
    for node_id, node in TOPOLOGY.items():
        if node_id == root_cause_id:
            color = "red"
            label = f"{node_id}\\n❌ 真因"
        elif node_id in cascade_set:
            color = "orange"
            label = f"{node_id}\\n⚠️ 監視不能"
        else:
            color = "lightgreen"
            label = node_id
        
        dot += f'    "{node_id}" [label="{label}", fillcolor={color}];\n'
    
    # エッジの追加
    for node_id, node in TOPOLOGY.items():
        if node.parent_id:
            dot += f'    "{node.parent_id}" -> "{node_id}";\n'
    
    dot += "}\n"
    return dot

# =====================================================
# メイン画面
# =====================================================

def main():
    st.title("🛡️ AIOps 障害分析システム")
    st.markdown("---")
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # APIキー設定
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            api_key = st.text_input("Google API Key", type="password")
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
        else:
            st.success("✅ APIキー設定済み")
        
        st.markdown("---")
        
        # シナリオ選択
        st.subheader("📋 障害シナリオ選択")
        scenario_keys = [k for k in SCENARIOS.keys() if SCENARIOS[k] is not None]
        selected_scenario = st.selectbox(
            "シナリオを選択",
            scenario_keys,
            index=0
        )
        
        st.markdown("---")
        
        # 分析実行ボタン
        if st.button("🚀 障害分析を実行", type="primary", use_container_width=True):
            if not api_key:
                st.error("❌ APIキーを設定してください")
            else:
                st.session_state.current_scenario = selected_scenario
                st.session_state.analysis_done = False
                st.session_state.remediation_executed = False
                st.session_state.health_check_done = False
                st.rerun()
        
        # リセットボタン
        if st.button("🔄 リセット", use_container_width=True):
            st.session_state.analysis_done = False
            st.session_state.current_scenario = None
            st.session_state.root_cause_result = None
            st.session_state.generated_log = ""
            st.session_state.remediation_executed = False
            st.session_state.health_check_done = False
            st.rerun()
    
    # メインコンテンツ
    if st.session_state.current_scenario and not st.session_state.analysis_done:
        perform_analysis(st.session_state.current_scenario, api_key)
    
    if st.session_state.analysis_done and st.session_state.root_cause_result:
        display_results(st.session_state.root_cause_result, api_key)

def perform_analysis(scenario: str, api_key: str):
    """障害分析を実行"""
    
    with st.spinner("🔍 障害分析を実行中..."):
        # 1. 対象ノード特定
        target_device_id = get_target_node_from_scenario(scenario)
        target_node = TOPOLOGY.get(target_device_id)
        
        if not target_node:
            st.error(f"❌ デバイス {target_device_id} が見つかりません")
            return
        
        # 2. 障害ログ生成
        st.info("📝 ステップ1: 障害ログを生成中...")
        time.sleep(0.5)
        
        log_result = run_diagnostic_simulation(scenario, target_node, api_key)
        generated_log = log_result.get("sanitized_log", "")
        st.session_state.generated_log = generated_log
        
        # 3. 大量アラーム生成
        st.info("🚨 ステップ2: アラームを生成中（50-200件）...")
        time.sleep(0.5)
        
        all_alarms = generate_massive_alarms(scenario, target_device_id)
        st.success(f"✅ {len(all_alarms)}件のアラームを生成しました")
        
        # 4. AIアラーム選別
        st.info("🎯 ステップ3: AIが重要なアラームを選別中...")
        time.sleep(1.0)
        
        critical_alarms = filter_critical_alarms(all_alarms, api_key)
        st.success(f"✅ {len(critical_alarms)}件の重要アラームを抽出しました")
        
        # 5. ログ検証
        st.info("🔬 ステップ4: ログを検証中...")
        time.sleep(0.5)
        
        verification = verify_log_content(generated_log)
        
        # 6. 因果推論
        st.info("🧠 ステップ5: 因果推論エンジンで真因を特定中...")
        time.sleep(1.0)
        
        engine = CausalInferenceEngine(TOPOLOGY)
        inference_result = engine.analyze_alarms(critical_alarms)
        
        # 7. LLM冗長性分析
        st.info("🤖 ステップ6: LLMで冗長性を分析中...")
        time.sleep(1.0)
        
        rca = LogicalRCA(TOPOLOGY)
        llm_analysis = rca.analyze(critical_alarms)
        
        # 8. カスケード影響分析
        st.info("📊 ステップ7: カスケード影響を分析中...")
        time.sleep(0.5)
        
        cascade_impact = get_cascade_impact(target_device_id)
        
        # 9. 復旧手順生成
        st.info("📋 ステップ8: 復旧手順を生成中...")
        time.sleep(1.0)
        
        remediation = generate_remediation_commands(
            scenario,
            llm_analysis[0] if llm_analysis else {},
            target_node,
            api_key
        )
        
        # 結果を保存
        st.session_state.root_cause_result = {
            "scenario": scenario,
            "target_device": target_device_id,
            "target_node": target_node,
            "all_alarms_count": len(all_alarms),
            "critical_alarms": critical_alarms,
            "inference_result": inference_result,
            "llm_analysis": llm_analysis,
            "verification": verification,
            "cascade_impact": cascade_impact,
            "remediation": remediation,
            "generated_log": generated_log
        }
        
        st.session_state.analysis_done = True
        st.rerun()

def display_results(result: Dict[str, Any], api_key: str):
    """分析結果を表示"""
    
    # 1. 真因特定の表示
    st.markdown("## 🎯 真因特定結果")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📉 ノイズ削減率",
            f"{((result['all_alarms_count'] - len(result['critical_alarms'])) / result['all_alarms_count'] * 100):.1f}%"
        )
    
    with col2:
        st.metric(
            "📨 総アラーム数",
            f"{result['all_alarms_count']}件"
        )
    
    with col3:
        st.metric(
            "✅ 選別後アラーム",
            f"{len(result['critical_alarms'])}件"
        )
    
    with col4:
        st.metric(
            "🎯 真因",
            "1件特定"
        )
    
    st.markdown("---")
    
    # 真因の大きな表示
    inference = result['inference_result']
    root_node = inference.root_cause_node
    
    if root_node:
        st.markdown(f"""
### 🚨 真因特定完了

<div style="background-color: #ff4444; padding: 20px; border-radius: 10px; color: white;">
<h2 style="color: white;">デバイス: {root_node.id}</h2>
<h3 style="color: white;">障害種別: {result['scenario']}</h3>
<p style="font-size: 18px;"><strong>影響度:</strong> {inference.severity}</p>
<p style="font-size: 18px;"><strong>確信度:</strong> {result['llm_analysis'][0]['prob'] * 100:.0f}%</p>
<p style="font-size: 16px;"><strong>理由:</strong> {inference.root_cause_reason}</p>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 2. チョイスされたアラーム表示
    with st.expander("🚨 チョイスされた重要アラーム", expanded=True):
        for i, alarm in enumerate(result['critical_alarms'], 1):
            severity_emoji = "🔴" if alarm.severity == "CRITICAL" else "🟡" if alarm.severity == "WARNING" else "⚪"
            st.markdown(f"{severity_emoji} **{i}.** `{alarm.device_id}` - {alarm.message} ({alarm.severity})")
    
    st.markdown("---")
    
    # 3. カスケード影響の説明
    cascade = result['cascade_impact']
    if cascade['count'] > 0:
        with st.expander("📊 カスケード障害の影響分析", expanded=True):
            st.markdown(cascade['reason'])
    
    st.markdown("---")
    
    # 4. トポロジー図
    st.markdown("## 🗺️ ネットワークトポロジー（色分け表示）")
    
    cascade_node_ids = [n.id for n in cascade['nodes']]
    topology_graph = generate_topology_graph(
        root_cause_id=result['target_device'],
        cascade_nodes=cascade_node_ids
    )
    
    st.graphviz_chart(topology_graph)
    
    st.markdown("""
**凡例:**
- 🔴 赤: 真因（根本原因）
- 🟠 オレンジ: 監視不能（カスケード影響）
- 🟢 緑: 正常稼働
""")
    
    st.markdown("---")
    
    # 5. 根本原因分析結果
    with st.expander("🔍 根本原因分析の詳細", expanded=True):
        st.markdown(f"""
**推論エンジン分析:**
- SOP Key: `{inference.sop_key}`
- 関連アラーム数: {len(inference.related_alarms)}件

**LLM分析結果:**
""")
        for analysis in result['llm_analysis']:
            st.json(analysis)
        
        st.markdown("**ログ検証結果（Ground Truth）:**")
        st.text(format_verification_report(result['verification']))
    
    st.markdown("---")
    
    # 6. 復旧手順
    st.markdown("## 📋 復旧手順")
    
    st.markdown(result['remediation'])
    
    st.markdown("---")
    
    # 7. 復旧措置ボタン
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔧 復旧措置を実行", type="primary", use_container_width=True):
            with st.spinner("復旧措置を実行中..."):
                time.sleep(2)
                st.session_state.remediation_executed = True
                st.rerun()
    
    with col2:
        if st.button("✅ 正常性確認", use_container_width=True):
            with st.spinner("正常性確認中..."):
                time.sleep(2)
                st.session_state.health_check_done = True
                st.rerun()
    
    # 復旧措置の結果
    if st.session_state.remediation_executed:
        st.success("✅ 復旧措置が完了しました")
        st.info("""
**実行内容:**
- 電源ユニットを交換しました
- デバイスを再起動しました
- インターフェースの状態を確認しました
""")
    
    # 正常性確認の結果
    if st.session_state.health_check_done:
        if result['scenario'] == "正常稼働":
            st.success("✅ すべてのデバイスが正常に稼働しています")
        else:
            # 正常性確認コマンドを生成・実行
            target_node = result['target_node']
            health_commands = generate_health_check_commands(target_node, api_key)
            
            st.success("✅ 正常性確認が完了しました")
            st.markdown(f"""
**確認結果:**
- デバイス {result['target_device']} は正常に復旧しました
- すべてのインターフェースが UP 状態です
- 配下の機器も正常に通信可能です

**実行したコマンド:**
{health_commands}
""")
    
    st.markdown("---")
    
    # 8. AIチャット欄
    st.markdown("## 💬 AIチャット（詳細確認）")
    
    user_question = st.text_input("質問を入力してください", placeholder="例: この障害の影響範囲を教えて")
    
    if user_question:
        with st.spinner("AIが回答を生成中..."):
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            context = f"""
あなたはネットワーク障害分析のエキスパートAIです。
以下の情報に基づいて、ユーザーの質問に答えてください。

【障害シナリオ】
{result['scenario']}

【真因デバイス】
{result['target_device']}

【分析結果】
{inference.root_cause_reason}

【カスケード影響】
{cascade['count']}台の機器が影響を受けています

【ユーザーの質問】
{user_question}
"""
            
            response = model.generate_content(context)
            st.markdown(f"**AI回答:**\n\n{response.text}")

# =====================================================
# 実行
# =====================================================
if __name__ == "__main__":
    main()
