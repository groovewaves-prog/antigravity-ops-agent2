import streamlit as st
import graphviz
import os
import google.generativeai as genai

# データ・ロジック・運用モジュールのインポート
from data import TOPOLOGY
from logic import CausalInferenceEngine, Alarm
# ★重要: 実機接続の代わりにスタブ(シミュレーション)関数を使用します
from network_ops import run_diagnostic_simulation

# --- ページ設定 ---
st.set_page_config(page_title="Antigravity Live", page_icon="⚡", layout="wide")

# --- 関数: トポロジー図の生成 (冗長構成対応) ---
def render_topology(alarms, root_cause_node):
    graph = graphviz.Digraph()
    graph.attr(rankdir='TB')
    graph.attr('node', shape='box', style='rounded,filled', fontname='Helvetica')
    
    alarmed_ids = {a.device_id for a in alarms}
    
    # ノード描画
    for node_id, node in TOPOLOGY.items():
        color = "#e8f5e9" # Default Green
        penwidth = "1"
        fontcolor = "black"
        label = f"{node_id}\n({node.type})"
        
        if root_cause_node and node_id == root_cause_node.id:
            color = "#ffcdd2" # Root Cause Red
            penwidth = "3"
            label += "\n[ROOT CAUSE]"
        elif node_id in alarmed_ids:
            color = "#fff9c4" # Alarm Yellow
        
        graph.node(node_id, label=label, fillcolor=color, color='black', penwidth=penwidth, fontcolor=fontcolor)
    
    # エッジ描画
    for node_id, node in TOPOLOGY.items():
        if node.parent_id:
            graph.edge(node.parent_id, node_id)
            
            # 親がHAグループの場合、相方からも線を引く
            parent_node = TOPOLOGY.get(node.parent_id)
            if parent_node and parent_node.redundancy_group:
                partners = [n.id for n in TOPOLOGY.values() 
                           if n.redundancy_group == parent_node.redundancy_group and n.id != parent_node.id]
                for partner_id in partners:
                    graph.edge(partner_id, node_id)
    return graph

# --- 関数: Config自動読み込み (IDベース) ---
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
st.title("⚡ Antigravity AI Agent (Live Demo)")

# APIキー取得
api_key = None
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = os.environ.get("GOOGLE_API_KEY")

# サイドバー
with st.sidebar:
    st.header("⚡ 運用モード選択")
    selected_scenario = st.radio(
        "シナリオ:", 
        ("正常稼働", "1. WAN全回線断", "2. FW片系障害", "3. L2SWサイレント障害", "4. [Live] Cisco実機診断")
    )
    
    st.markdown("---")
    if api_key:
        st.success("API Connected")
    else:
        st.warning("API Key Missing")
        user_key = st.text_input("Google API Key", type="password")
        if user_key: api_key = user_key

# セッション状態管理
if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = "正常稼働"
    st.session_state.messages = []
    st.session_state.chat_session = None 
    st.session_state.live_result = None

# シナリオ変更時のリセット処理
if st.session_state.current_scenario != selected_scenario:
    st.session_state.current_scenario = selected_scenario
    st.session_state.messages = []
    st.session_state.chat_session = None
    st.session_state.live_result = None
    st.rerun()

# --- アラーム生成ロジック (シミュレーション用) ---
alarms = []
if selected_scenario == "1. WAN全回線断":
    alarms = [
        Alarm("WAN_ROUTER_01", "Interface Down", "CRITICAL"),
        Alarm("FW_01_PRIMARY", "Gateway Unreachable", "WARNING"),
        Alarm("FW_01_SECONDARY", "Gateway Unreachable", "WARNING"),
        Alarm("CORE_SW_01", "Uplink Down", "WARNING"),
        Alarm("AP_01", "Unreachable", "CRITICAL")
    ]
elif selected_scenario == "2. FW片系障害":
    alarms = [Alarm("FW_01_PRIMARY", "Heartbeat Loss", "WARNING")]
elif selected_scenario == "3. L2SWサイレント障害":
    alarms = [Alarm("AP_01", "Connection Lost", "CRITICAL"), Alarm("AP_02", "Connection Lost", "CRITICAL")]

# 推論実行 (通常シナリオの場合)
root_cause = None
inference_result = None
reason = ""

if alarms:
    engine = CausalInferenceEngine(TOPOLOGY)
    inference_result = engine.analyze_alarms(alarms)
    root_cause = inference_result.root_cause_node
    reason = inference_result.root_cause_reason

# --- メイン画面レイアウト ---
col1, col2 = st.columns([1, 1])

# 左カラム：トポロジー ＆ 自律調査UI
with col1:
    st.subheader("Network Status")
    st.graphviz_chart(render_topology(alarms, root_cause), use_container_width=True)
    
    # アラート表示
    if root_cause:
        st.markdown(
            f'<div style="color: #d32f2f; font-weight: bold; font-size: 15px; background-color: #fdecea; padding: 10px; border-radius: 5px;">'
            f'🚨 緊急アラート：{root_cause.id} ダウン'
            f'</div>', 
            unsafe_allow_html=True
        )
        st.caption(f"理由: {reason}")
    
    # Liveモードの場合のUI
    is_live_mode = (selected_scenario == "4. [Live] Cisco実機診断")
    
    if is_live_mode or root_cause: # 障害時は常に調査ボタンを出しても良いが、今回はLiveモード強調
        st.markdown("---")
        st.info("🛠 **自律調査エージェント**")
        
        # ボタン: 診断実行 (スタブ関数を呼び出し)
        if st.button("🚀 診断実行 (Simulation)", type="primary"):
            if not api_key:
                st.error("API Key Required")
            else:
                with st.status("Agent Operating...", expanded=True) as status:
                    st.write("🔌 Initiating connection simulation...")
                    # スタブ関数を呼び出して結果を取得
                    res = run_diagnostic_simulation(selected_scenario)
                    st.session_state.live_result = res
                    
                    if res["status"] == "SUCCESS":
                        st.write("✅ Data retrieved.")
                        status.update(label="Complete!", state="complete", expanded=False)
                    else:
                        st.write("❌ Connection Failed (As expected in failure scenario).")
                        status.update(label="Target Unreachable", state="error", expanded=False)

        # 診断結果の表示
        if st.session_state.live_result:
            res = st.session_state.live_result
            if res["status"] == "SUCCESS":
                # セキュリティバナー (アプローチ1)
                st.success("🛡️ **Data Sanitized**: パスワード・IPアドレスをマスク処理しました。")
                
                with st.expander("📄 取得ログ (Sanitized View)", expanded=True):
                    st.code(res["sanitized_log"], language="text")
            else:
                st.error(f"診断結果: {res['error']}")
                st.caption("※エージェントはこの接続エラー自体を『診断情報』として利用します。")

# 右カラム：AIチャット (スクロール対応)
with col2:
    st.subheader("AI Analyst Report")

    # APIキーチェック
    if not api_key:
        st.error("APIキーを設定してください")
        st.stop()

    # チャットセッション初期化 (初回のみ)
    # Live結果がある場合、または推論結果がある場合に起動
    should_start_chat = (st.session_state.chat_session is None) and (selected_scenario != "正常稼働")
    
    if should_start_chat:
        genai.configure(api_key=api_key)
        
        # 設定: Gemini 2.0 Flash, 温度0
        generation_config = {
            "temperature": 0.0,
            "max_output_tokens": 1500,
        }
        model = genai.GenerativeModel("gemini-2.0-flash", generation_config=generation_config)
        
        # --- プロンプト構築 ---
        system_prompt = ""
        
        # A. Live診断結果がある場合 (Liveモード優先)
        if st.session_state.live_result:
            live_data = st.session_state.live_result
            system_prompt = f"""
            あなたはネットワークエンジニアです。以下の診断結果に基づき、トラブルシューティングの経緯を報告してください。

            【診断入力データ】
            ステータス: {live_data['status']}
            詳細/ログ: {live_data.get('sanitized_log') or live_data.get('error')}
            推論された原因: {reason if reason else "実機調査モード"}

            【出力要件】
            以下のフォーマットで出力すること。
            
            ### 🛠 ネクストアクション実行レポート
            
            **1. データ保全と接続確認:**
            接続試行およびログ取得を実施。
            → **結果: {live_data['status']}** (🛡️ 機密情報はフィルタリング済み)
            
            **2. 詳細分析:**
            [接続できた場合はログ内容（Config/Interface）の分析、エラーの場合は要因推測]
            → [分析結果]
            
            **3. 物理/インターフェース確認:**
            [状況に応じた推論]
            → [分析結果]
            
            ---
            **最終判定:** [結論]
            """

        # B. 通常の推論モードの場合 (Live診断前)
        elif root_cause:
            # Config読み込み
            config_content = load_config_by_id(root_cause.id)
            
            system_prompt = f"""
            あなたはAIOpsエージェントです。以下の障害について報告してください。
            
            根本原因: {root_cause.id} ({root_cause.type})
            理由: {inference_result.root_cause_reason}
            """
            
            if config_content:
                system_prompt += f"\n【Configあり】\n{config_content}\n上記設定に基づき、具体的な確認コマンドを提案してください。"
            else:
                system_prompt += "\n【Configなし】\n一般的な復旧手順を提示してください。"
            
            system_prompt += "\nフォーマット: 緊急度(絵文字)、状況要約、推奨SOPの順で出力。"

        # チャット開始
        if system_prompt:
            history = [{"role": "user", "parts": [system_prompt]}]
            chat = model.start_chat(history=history)
            
            try:
                # 最初の分析を実行
                with st.spinner("Gemini is analyzing..."):
                    response = chat.send_message("レポートを作成してください。")
                    st.session_state.chat_session = chat
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"Error: {e}")

    # --- チャットUIの表示 (スクロールコンテナ) ---
    chat_container = st.container(height=600)
    
    with chat_container:
        # 履歴表示
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 入力欄 (コンテナの外に配置して固定)
    if prompt := st.chat_input("AIエージェントに指示..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # ユーザー入力を即時表示
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # AI応答
        if st.session_state.chat_session:
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            res = st.session_state.chat_session.send_message(prompt)
                            st.markdown(res.text)
                            st.session_state.messages.append({"role": "assistant", "content": res.text})
                        except Exception as e:
                            st.error(f"Error: {e}")
