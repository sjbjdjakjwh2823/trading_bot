import torch
import pandas as pd
import numpy as np
import pandas_ta as ta
from supabase import create_client
from sklearn.preprocessing import MinMaxScaler
import re
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 1. 설정
# ==========================================
INITIAL_BALANCE = 10000.0 
LEVERAGE = 10               # 레버리지 10배
FEE = 0.0004                # 수수료 0.04%

URL = ""
RAW_KEY = ""
KEY = re.sub(r'[^\x00-\x7F]+', '', RAW_KEY).strip()
supabase = create_client(URL, KEY)

class PricePredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(PricePredictor, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = torch.nn.Linear(hidden_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

def fetch_all_data(table_name):
    print(f"📡 {table_name} 로드 중...")
    all_data = []
    last_timestamp = 0
    while True:
        res = supabase.table(table_name).select("*").gt("timestamp", last_timestamp).order("timestamp").limit(1000).execute()
        if not res.data: break
        all_data.extend(res.data)
        last_timestamp = res.data[-1]['timestamp']
        if len(all_data) > 60000: break # 충분한 데이터 확보
    df = pd.DataFrame(all_data)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.sort_values('datetime').reset_index(drop=True)

# ==========================================
# 2. 백테스트 실행 (8:2 분리 적용)
# ==========================================
def run_backtest():
    df_15m = fetch_all_data("trading_data_15m")
    df_1h = fetch_all_data("trading_data_1h")
    df_4h = fetch_all_data("trading_data_4h")
    df_1d = fetch_all_data("trading_data_1d")

    print("📈 지표 계산 및 데이터 병합...")
    for df, sfx in [(df_15m, ""), (df_1h, "_1h"), (df_4h, "_4h"), (df_1d, "_1d")]:
        df[f'rsi{sfx}'] = ta.rsi(df['close'], length=14)
    
    bb = ta.bbands(df_15m['close'], length=20)
    df_15m['bb_u'], df_15m['bb_l'] = bb.iloc[:, 2], bb.iloc[:, 0]
    macd = ta.macd(df_15m['close'])
    df_15m['macd'], df_15m['macd_s'] = macd.iloc[:, 0], macd.iloc[:, 1]

    merged = pd.merge_asof(df_15m, df_1h[['datetime', 'rsi_1h']], on='datetime', direction='backward')
    merged = pd.merge_asof(merged, df_4h[['datetime', 'rsi_4h']], on='datetime', direction='backward')
    final_df = pd.merge_asof(merged, df_1d[['datetime', 'rsi_1d']], on='datetime', direction='backward').dropna()

    # [핵심] 8:2 데이터 분리
    # 전체 데이터의 마지막 20%만 똑 떼어내서 테스트합니다. (AI가 본 적 없는 미래라고 가정)
    split_idx = int(len(final_df) * 0.8)
    test_df = final_df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"🕵️ 전체 {len(final_df)}개 중 뒤쪽 20%인 {len(test_df)}개 데이터로 '진짜 실력' 검증 시작...")

    features = ['close', 'volume', 'rsi', 'bb_u', 'bb_l', 'macd', 'macd_s', 'rsi_1h', 'rsi_4h', 'rsi_1d']
    data_x = test_df[features].values
    prices = test_df['close'].values
    times = test_df['datetime'].values

    scaler = MinMaxScaler()
    data_x_scaled = scaler.fit_transform(data_x)
    
    X_list = []
    for i in range(len(data_x_scaled) - 60):
        X_list.append(data_x_scaled[i:i+60])
    X_tensor = torch.FloatTensor(np.array(X_list))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PricePredictor(input_dim=10, hidden_dim=64, num_layers=2).to(device)
    model.load_state_dict(torch.load("trading_model.pth", map_location=device))
    model.eval()

    print("🤖 AI 시뮬레이션 가동 중...")
    predictions = []
    loader = DataLoader(TensorDataset(X_tensor), batch_size=512)
    with torch.no_grad():
        for batch in loader:
            predictions.extend(model(batch[0].to(device)).cpu().numpy().flatten())

    # ==========================================
    # 3. 자산 시뮬레이션 루프
    # ==========================================
    balance = INITIAL_BALANCE
    position = 0 
    trade_count = 0
    
    for i in range(len(predictions)):
        prob = predictions[i]
        curr_price = prices[i + 60]
        next_price = prices[i + 61] if i + 61 < len(prices) else curr_price
        
        signal = 0
        if prob >= 0.55: signal = 1
        elif prob <= 0.45: signal = -1

        if signal != position:
            if position != 0: balance -= balance * FEE 
            if signal != 0: 
                balance -= balance * FEE 
                trade_count += 1
            position = signal

        if position == 1:
            balance += balance * ((next_price - curr_price) / curr_price) * LEVERAGE
        elif position == -1:
            balance += balance * ((curr_price - next_price) / curr_price) * LEVERAGE

        if balance <= 0:
            print(f"💀 [{times[i+60]}] 마진콜 발생!")
            balance = 0
            break

    print("\n" + "="*50)
    print(f"📊 정직한 백테스팅 결과 (최근 20% 데이터)")
    print("-" * 50)
    print(f"💰 시작 자산: ${INITIAL_BALANCE:,.2f}")
    print(f"💰 최종 자산: ${balance:,.2f}")
    print(f"📈 총 수익률: {((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100:.2f}%")
    print(f"🔄 총 매매 횟수: {trade_count}회")
    print("="*50)

if __name__ == "__main__":
    run_backtest()