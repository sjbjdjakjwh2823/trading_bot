import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from supabase import create_client, Client
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import os

# ==========================================
# 1. 환경 설정 (본인의 정보로 수정)
# ==========================================
SUPABASE_URL = ""
SUPABASE_KEY = ""
TABLE_NAME = "trading_data_1d"  # 슈퍼베이스 테이블 이름


# ==========================================
# 2. 슈퍼베이스 데이터 로드 함수
# ==========================================
def fetch_data_from_supabase():
    print("🌐 슈퍼베이스에서 데이터를 가져오는 중...")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 전체 데이터를 가져옵니다 (필요시 .range()로 조절 가능)
    response = supabase.table(TABLE_NAME).select("*").order("timestamp").execute()
    df = pd.DataFrame(response.data)

    # 숫자형 변환 및 정렬
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'ma50', 'ma200']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()  # 결측치 제거
    return df


# ==========================================
# 3. 데이터 전처리 및 시퀀스 생성
# ==========================================
def prepare_data(df, window_size=24):
    features = ['open', 'high', 'low', 'close', 'volume', 'ma50', 'ma200']

    # [라벨링] 24시간 내 3% 수익 발생 시 Target=1 (파동 포착)
    df['target'] = 0
    close_prices = df['close'].values
    high_prices = df['high'].values
    for i in range(len(df) - 24):
        future_max = np.max(high_prices[i + 1: i + 25])
        if (future_max - close_prices[i]) / close_prices[i] >= 0.03:
            df.at[df.index[i], 'target'] = 1

    data = df[features].values
    target = df['target'].values

    # 정규화 (MinMax 스케일링)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # LSTM용 시퀀스 생성 (과거 24시간 -> 현재 예측)
    X, y = [], []
    for i in range(len(data_scaled) - window_size):
        X.append(data_scaled[i: i + window_size])
        y.append(target[i + window_size])

    return torch.tensor(np.array(X), dtype=torch.float32), \
        torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1), \
        scaler


# ==========================================
# 4. LSTM 모델 구조 정의
# ==========================================
class WaveLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(WaveLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])  # 마지막 시점의 은닉 상태 사용
        return self.sigmoid(out)


# ==========================================
# 5. 메인 실행 루틴 (학습)
# ==========================================
if __name__ == "__main__":
    # 1. 데이터 가져오기
    try:
        df_raw = fetch_data_from_supabase()
        X_data, y_data, scaler = prepare_data(df_raw)

        # 데이터셋 분리
        train_size = int(len(X_data) * 0.8)
        train_X, train_y = X_data[:train_size], y_data[:train_size]

        loader = DataLoader(TensorDataset(train_X, train_y), batch_size=32, shuffle=True)

        # 2. 모델 설정
        model = WaveLSTM(input_dim=7, hidden_dim=64, num_layers=2)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 3. 학습
        print("🚀 LSTM 모델 학습 시작 (Supabase Data)...")
        for epoch in range(100):
            model.train()
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/100], Loss: {total_loss / len(loader):.4f}")

        # 4. 모델 저장
        torch.save(model.state_dict(), "wave_lstm_model.pth")
        print("✅ 학습 완료! 'wave_lstm_model.pth' 저장됨.")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")