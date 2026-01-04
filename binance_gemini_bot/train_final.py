import pandas as pd
import numpy as np
import pandas_ta as ta
import torch
import torch.nn as nn
import re
from supabase import create_client
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 1. 설정 (사용자 정보 입력)
# ==========================================
URL = ""
RAW_KEY = "" # 슈퍼베이스 'anon' 또는 'service_role' 키

# 키값에서 혹시 모를 특수문자/공백 제거
KEY = re.sub(r'[^\x00-\x7F]+', '', RAW_KEY).strip()
supabase = create_client(URL, KEY)

# ==========================================
# 2. 데이터 로딩 함수 (1,000개 제한 해제)
# ==========================================
def fetch_all_data(table_name):
    print(f"📡 {table_name} 모든 데이터 로드 중...")
    all_data = []
    last_timestamp = 0
    
    while True:
        res = supabase.table(table_name).select("*")\
            .gt("timestamp", last_timestamp)\
            .order("timestamp")\
            .limit(1000).execute()
        
        if not res.data:
            break
            
        all_data.extend(res.data)
        last_timestamp = res.data[-1]['timestamp']
        
        if len(all_data) % 10000 == 0:
            print(f"   -> {len(all_data)}개 완료...")

    df = pd.DataFrame(all_data)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df.sort_values('datetime').reset_index(drop=True)

# ==========================================
# 3. 데이터 전처리 및 병합
# ==========================================
def prepare_data():
    # 데이터 가져오기
    df_15m = fetch_all_data("trading_data_15m")
    df_1h = fetch_all_data("trading_data_1h")
    df_4h = fetch_all_data("trading_data_4h")
    df_1d = fetch_all_data("trading_data_1d")

    print("📈 보조지표 계산 및 병합 중...")
    
    # 각 시간대별 RSI 계산
    for df, sfx in [(df_15m, ""), (df_1h, "_1h"), (df_4h, "_4h"), (df_1d, "_1d")]:
        df[f'rsi{sfx}'] = ta.rsi(df['close'], length=14)

    # 15분봉 전용 지표 (BB, MACD)
    bb = ta.bbands(df_15m['close'], length=20, std=2)
    df_15m['bb_u'] = bb.iloc[:, 2] # Upper
    df_15m['bb_l'] = bb.iloc[:, 0] # Lower
    
    macd = ta.macd(df_15m['close'])
    df_15m['macd'] = macd.iloc[:, 0]
    df_15m['macd_s'] = macd.iloc[:, 1]

    # 멀티 타임프레임 병합 (시간 기준 정렬)
    merged = pd.merge_asof(df_15m, df_1h[['datetime', 'rsi_1h']], on='datetime', direction='backward')
    merged = pd.merge_asof(merged, df_4h[['datetime', 'rsi_4h']], on='datetime', direction='backward')
    final_df = pd.merge_asof(merged, df_1d[['datetime', 'rsi_1d']], on='datetime', direction='backward')

    # Target 생성: 4시간 뒤 가격이 올랐으면 1, 아니면 0
    final_df['target'] = (final_df['close'].shift(-16) > final_df['close']).astype(int)
    
    # 결측치 제거 및 데이터 제한 (최근 5만개만 사용하여 학습 속도 최적화)
    final_df = final_df.dropna().iloc[-50000:]
    
    print(f"✅ 최종 학습 데이터 준비 완료: {len(final_df)}개")
    return final_df

# ==========================================
# 4. LSTM 모델 정의
# ==========================================
class PricePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(PricePredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# ==========================================
# 5. 실행 메인 루프
# ==========================================
if __name__ == "__main__":
    try:
        df = prepare_data()
        
        # 1. 특성 선택 및 데이터 준비
        features = ['close', 'volume', 'rsi', 'bb_u', 'bb_l', 'macd', 'macd_s', 'rsi_1h', 'rsi_4h', 'rsi_1d']
        data_x = df[features].values
        data_y = df['target'].values

        # 2. 스케일링
        scaler = MinMaxScaler()
        data_x_scaled = scaler.fit_transform(data_x)

        # 3. 시퀀스 생성 함수
        def create_sequences(data, target, seq_length):
            x, y = [], []
            for i in range(len(data) - seq_length):
                x.append(data[i:i+seq_length])
                y.append(target[i+seq_length])
            return np.array(x), np.array(y)

        X, y = create_sequences(data_x_scaled, data_y, 60)
        
        # 4. 텐서 변환 및 데이터로더 설정 (이 부분이 핵심!)
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).view(-1, 1)
        
        # 데이터를 64개씩 쪼개서 모델에 넣습니다 (메모리 폭주 방지)
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        # 5. 모델 설정
        # 만약 NVIDIA 그래픽카드가 있다면 cuda를 사용합니다.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ 현재 사용 장치: {device}")
        
        model = PricePredictor(input_dim=len(features), hidden_dim=64, num_layers=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # 6. 학습 루프 (배치 단위 학습)
        print(f"🚀 학습 시작... (총 {len(train_loader)}개 배치)")
        for epoch in range(1, 11):
            model.train()
            epoch_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            print(f"✅ Epoch [{epoch}/10], 평균 Loss: {avg_loss:.4f}")

        # 7. 모델 저장
        torch.save(model.state_dict(), "trading_model.pth")
        print("💾 모델 저장 완료: trading_model.pth")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")