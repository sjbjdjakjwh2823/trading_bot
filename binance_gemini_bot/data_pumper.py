import ccxt
import pandas as pd
from supabase import create_client
import time
from datetime import datetime

# ==========================================
# 설정
# ==========================================
SUPABASE_URL = ""
SUPABASE_KEY = ""

SYMBOL = 'BTC/USDT'
TIMEFRAMES = {
    '15m': 'trading_data_15m',
    '1h': 'trading_data_1h',
    '4h': 'trading_data_4h',
    '1d': 'trading_data_1d'
}

# 거래소 초기화
exchange = ccxt.binance({
    'enableRateLimit': True,
})

# Supabase 초기화
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_and_save_ohlcv(timeframe, table_name, limit=1000):
    """거래소에서 OHLCV 데이터를 가져와서 Supabase에 저장"""
    try:
        print(f"\n📊 {timeframe} 데이터 수집 중...")
        
        # 거래소에서 데이터 가져오기
        ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe, limit=limit)
        
        if not ohlcv:
            print(f"⚠️ {timeframe} 데이터가 없습니다.")
            return
        
        # DataFrame으로 변환
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # MA 계산 (필요한 경우)
        df['ma50'] = df['close'].rolling(50).mean()
        df['ma200'] = df['close'].rolling(200).mean()
        
        # 결측치 제거
        df = df.dropna().reset_index(drop=True)
        
        if len(df) == 0:
            print(f"⚠️ {timeframe} 처리할 데이터가 없습니다.")
            return
        
        # Supabase 형식에 맞게 변환
        records = []
        for _, row in df.iterrows():
            record = {
                'timestamp': int(row['timestamp']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'ma50': float(row['ma50']) if pd.notna(row['ma50']) else None,
                'ma200': float(row['ma200']) if pd.notna(row['ma200']) else None
            }
            records.append(record)
        
        # 기존 데이터 확인 (중복 방지)
        if records:
            # 마지막 타임스탬프 확인
            last_record = supabase.table(table_name).select("timestamp").order("timestamp", desc=True).limit(1).execute()
            
            if last_record.data:
                last_timestamp = last_record.data[0]['timestamp']
                # 마지막 타임스탬프 이후의 데이터만 추가
                records = [r for r in records if r['timestamp'] > last_timestamp]
            
            if records:
                # 배치로 삽입 (한 번에 100개씩)
                batch_size = 100
                for i in range(0, len(records), batch_size):
                    batch = records[i:i+batch_size]
                    supabase.table(table_name).insert(batch).execute()
                    print(f"   ✅ {len(batch)}개 데이터 저장 완료 (총 {min(i+batch_size, len(records))}/{len(records)})")
                
                print(f"✅ {timeframe} 데이터 저장 완료: {len(records)}개")
            else:
                print(f"ℹ️ {timeframe} 새로운 데이터가 없습니다.")
        else:
            print(f"⚠️ {timeframe} 저장할 데이터가 없습니다.")
            
    except Exception as e:
        print(f"❌ {timeframe} 오류 발생: {e}")

def pump_data_multi():
    """모든 시간대의 데이터를 수집하고 저장"""
    print("=" * 60)
    print("🚀 거래소 데이터 수집 시작")
    print("=" * 60)
    
    for timeframe, table_name in TIMEFRAMES.items():
        try:
            fetch_and_save_ohlcv(timeframe, table_name)
            time.sleep(1)  # API 호출 제한 고려
        except Exception as e:
            print(f"❌ {timeframe} 처리 중 오류: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("🎉 모든 시간대 데이터 수집 완료!")
    print("=" * 60)

if __name__ == "__main__":
    pump_data_multi()


