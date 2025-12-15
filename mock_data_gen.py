import pandas as pd
import random

# 生成するデータ数
NUM_SAMPLES = 5000

# ■ 世界の法則（シナリオ定義）
SCENARIOS = [
    # 1. WANルーター物理故障
    {
        "root_cause_id": "WAN_ROUTER_01",
        "root_cause_type": "Hardware/Physical",
        "weight": 0.2, 
        "probabilities": {
            ("alarm", "BGP Flapping"): 0.70,
            ("log", "Interface Down"): 0.95,
            ("ping", "NG"): 0.90,
            ("log", "Power Fail"): 0.10
        }
    },
    # 2. WANルーター設定ミス
    {
        "root_cause_id": "WAN_ROUTER_01",
        "root_cause_type": "Config/Software",
        "weight": 0.3,
        "probabilities": {
            ("alarm", "BGP Flapping"): 0.85,
            ("log", "Interface Down"): 0.05,
            ("ping", "NG"): 0.30,
            ("log", "Config Error"): 0.80
        }
    },
    # 3. FWハードウェア障害
    {
        "root_cause_id": "FW_01_PRIMARY",
        "root_cause_type": "Hardware/Physical",
        "weight": 0.1,
        "probabilities": {
            ("alarm", "HA Failover"): 0.90,
            ("ping", "NG"): 0.50,
            ("log", "Power Fail"): 0.80
        }
    },
    # 4. L2スイッチ/FAN故障 (★新規追加)
    {
        "root_cause_id": "L2_SW", # 汎用的なIDとして学習させる
        "root_cause_type": "Hardware/Fan",
        "weight": 0.15,
        "probabilities": {
            ("alarm", "Fan Fail"): 0.95,      # FAN故障なら必ずFANアラームが出る
            ("log", "High Temperature"): 0.60, # 温度上昇も伴うことが多い
            ("ping", "NG"): 0.05,             # FAN故障だけなら通信は切れないことが多い
            ("log", "System Warning"): 0.80
        }
    },
    # 5. 外部ISP障害（ノイズ）
    {
        "root_cause_id": "External_ISP",
        "root_cause_type": "Network",
        "weight": 0.2,
        "probabilities": {
            ("alarm", "BGP Flapping"): 0.60,
            ("log", "Interface Down"): 0.01,
            ("ping", "NG"): 0.80
        }
    }
]

def generate_mock_data():
    data = []
    print(f"Generating {NUM_SAMPLES} training samples based on World Model...")
    
    for _ in range(NUM_SAMPLES):
        scenario = random.choices(SCENARIOS, weights=[s["weight"] for s in SCENARIOS])[0]
        
        # IDの揺らぎを持たせる（L2_SWなら L2_SW_01, L2_SW_02 など）
        r_id = scenario['root_cause_id']
        if r_id == "L2_SW":
            r_id = random.choice(["L2_SW_01", "L2_SW_02"])
            
        root_key = f"{r_id}::{scenario['root_cause_type']}"
        
        for (ev_type, ev_val), prob in scenario["probabilities"].items():
            if random.random() < prob:
                data.append({
                    "RootCause": root_key,
                    "EvidenceType": ev_type,
                    "EvidenceValue": ev_val
                })
        
        if random.random() < 0.05:
            data.append({
                "RootCause": root_key,
                "EvidenceType": "log",
                "EvidenceValue": "Unknown Error"
            })

    df = pd.DataFrame(data)
    df.to_csv("training_data.csv", index=False)
    print(f"✅ Saved 'training_data.csv' ({len(df)} records).")

if __name__ == "__main__":
    generate_mock_data()
