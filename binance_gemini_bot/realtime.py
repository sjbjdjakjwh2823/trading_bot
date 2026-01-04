import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import threading
import ccxt
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os
import sys

# ====================================================
# [설정] OpenMP 에러 방지 (윈도우 충돌 방지용)
# ====================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ====================================================
# [사용자 설정] API 키 입력 (여기에 본인의 키를 넣으세요!)
# ====================================================
API_KEY = ""
SECRET_KEY = ""

SYMBOL = 'BTC/USDT:USDT'
TIMEFRAME_MAIN = '1h'
TIMEFRAME_SUB = '15m'
MODEL_FILENAME = "wave_lstm_model.pth" # 같은 폴더에 있어야 함

# 전역 변수
is_running = False
bot_thread = None
curr_leverage = 10
curr_balance_pct = 50.0
curr_tp_pct = 10.0
curr_sl_pct = 5.0

# 거래소 객체 생성
exchange = ccxt.gate({
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

# AI 모델 정의
class WaveLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(WaveLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return self.sigmoid(out)

def log(message):
    """ 로그창에 메시지 출력 """
    timestamp = time.strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] {message}\n"
    print(full_msg.strip()) # 터미널에도 출력
    try:
        log_area.configure(state='normal')
        log_area.insert(tk.END, full_msg)
        log_area.see(tk.END)
        log_area.configure(state='disabled')
    except:
        pass

def load_model():
    """ 모델 파일 로드 (강력한 탐색 기능) """
    try:
        # PyInstaller로 패키징된 경우 경로 처리
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
            exe_dir = os.path.dirname(sys.executable)
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
            exe_dir = base_path
        
        # 여러 경로에서 모델 파일 찾기
        possible_paths = [
            os.path.join(base_path, MODEL_FILENAME),
            os.path.join(exe_dir, MODEL_FILENAME),
            MODEL_FILENAME,
            os.path.join(os.getcwd(), MODEL_FILENAME)
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        device = torch.device('cpu')
        model = WaveLSTM(7, 64, 2).to(device)
        
        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            log(f"✅ AI Model Loaded: {model_path}")
        else:
            log(f"❌ Error: '{MODEL_FILENAME}'을 찾을 수 없습니다.")
            return None, None
            
        model.eval()
        return model, device
    except Exception as e:
        log(f"❌ Model Error: {e}")
        return None, None

def update_dashboard():
    """ 화면 갱신 (가격, 잔고 등) """
    if not root.winfo_exists(): return
    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        current_price = ticker['last']
        lbl_price_val.config(text=f"${current_price:,.2f}")

        balance = exchange.fetch_balance()
        free_usdt = balance['USDT']['free']
        lbl_balance_val.config(text=f"{free_usdt:.2f} USDT")
        
        positions = exchange.fetch_positions([SYMBOL])
        my_pos = None
        for pos in positions:
            if pos['symbol'] == SYMBOL:
                my_pos = pos
                break
        
        if my_pos and float(my_pos['contracts']) != 0:
            size = float(my_pos['contracts'])
            side = "LONG" if size > 0 else "SHORT"
            entry = float(my_pos['entryPrice'])
            
            # PNL 계산
            raw_pnl = (current_price - entry) / entry * 100
            if size < 0: raw_pnl = -raw_pnl
            lev = int(entry_leverage.get()) if entry_leverage.get() else 10
            real_pnl = raw_pnl * lev
            
            color = "#00ff00" if real_pnl >= 0 else "#ff0000"
            lbl_pos_val.config(text=f"{side} ({real_pnl:.2f}%)", fg=color)
            lbl_entry_val.config(text=f"${entry:,.2f}")
        else:
            lbl_pos_val.config(text="No Position", fg="#888888")
            lbl_entry_val.config(text="-")
    except Exception as e:
        pass
    root.after(1500, update_dashboard)

def get_my_position_info():
    try:
        positions = exchange.fetch_positions([SYMBOL])
        for pos in positions:
            if pos['symbol'] == SYMBOL:
                size = float(pos.get('contracts', 0) or pos.get('size', 0) or 0)
                if size == 0: return 0.0, 0.0, 0.0
                entry = float(pos.get('entryPrice', 0.0))
                curr = float(exchange.fetch_ticker(SYMBOL)['last'])
                pnl = (curr - entry) / entry * 100 * curr_leverage
                if size < 0: pnl = (entry - curr) / entry * 100 * curr_leverage
                return size, entry, pnl
        return 0.0, 0.0, 0.0
    except: return 0.0, 0.0, 0.0

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    return 100 - (100 / (1 + gain / loss))

def check_entry_signal(target_side):
    try:
        log(f"🔎 Analyzing {target_side}...")
        candles_1h = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME_MAIN, limit=50)
        candles_15m = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME_SUB, limit=50)
        df_1h = pd.DataFrame(candles_1h, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        df_15m = pd.DataFrame(candles_15m, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        
        rsi_1h = calculate_rsi(df_1h['c']).iloc[-1]
        rsi_15m = calculate_rsi(df_15m['c']).iloc[-1]
        close_15m = df_15m['c'].iloc[-1]
        ma20_15m = df_15m['c'].rolling(20).mean().iloc[-1]

        if target_side == 'LONG':
            if (rsi_1h < 70) and (rsi_15m < 45 or close_15m > ma20_15m): return True
        elif target_side == 'SHORT':
            if (rsi_1h > 30) and (rsi_15m > 55 or close_15m < ma20_15m): return True
        log("✋ Wait Signal (보조지표 조건 미충족)...")
        return False
    except: return False

def get_1h_ai_decision(model, device):
    try:
        candles = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME_MAIN, limit=300)
        df = pd.DataFrame(candles, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        df['ma50'] = df['c'].rolling(50).mean()
        df['ma200'] = df['c'].rolling(200).mean()
        df = df.dropna().reset_index(drop=True)
        if len(df) < 24: return 'WAIT'
        
        recent = df.iloc[-24:][['o', 'h', 'l', 'c', 'v', 'ma50', 'ma200']].values
        scaler = MinMaxScaler()
        data = scaler.fit_transform(recent)
        tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad(): pred = model(tensor).item()
        
        # 로그에 점수 출력
        log(f"🤖 AI Score: {pred:.4f}")
        
        if pred >= 0.7: return 'LONG'
        elif pred <= 0.3: return 'SHORT'  # 0.3 이하면 숏
        else: return 'WAIT'
    except: return 'WAIT'

def set_leverage(leverage):
    """거래소에 레버리지 설정"""
    try:
        exchange.set_leverage(leverage, SYMBOL)
        return True
    except:
        # 실패시 다른 방법 시도 등 생략 (Execute Order에서 파라미터로 처리됨)
        return False

def execute_order(side, reduce_only=False):
    try:
        action = "Close" if reduce_only else "Open"
        log(f"⚡ {action}: {side} Order...")
        
        current_leverage = int(entry_leverage.get()) if entry_leverage.get() else curr_leverage
        current_balance_pct = float(entry_balance.get()) if entry_balance.get() else curr_balance_pct
        
        if not reduce_only:
            set_leverage(current_leverage)
        
        exchange.load_markets()
        market = exchange.market(SYMBOL)
        contract_size = market.get('contractSize', 0.0001)
        final_contracts = 0

        if reduce_only:
            my_size, _, _ = get_my_position_info()
            final_contracts = abs(int(my_size))
        else:
            balance = exchange.fetch_balance()
            # ========================================================
            # [안전장치] 수수료 확보를 위해 잔고의 98%만 사용
            # ========================================================
            available_balance = balance['USDT']['free'] * 0.98 
            margin = available_balance * (current_balance_pct / 100)
            
            # 최소 주문 금액 보정 ($6 미만이면 $6으로 맞춤 - 잔고 충분할 시)
            if margin < 6: margin = 6
            if margin > available_balance: margin = available_balance # 그래도 없으면 전액

            pos_amt = margin * current_leverage
            current_price = exchange.fetch_ticker(SYMBOL)['last']
            final_contracts = int((pos_amt / current_price) / contract_size)
        
        if final_contracts < 1:
            log("❌ 잔고 부족 또는 주문 수량 부족")
            return
            
        params = {'reduceOnly': True} if reduce_only else {}
        params['leverage'] = current_leverage
        
        if side == 'buy': 
            exchange.create_market_buy_order(SYMBOL, final_contracts, params)
        elif side == 'sell': 
            exchange.create_market_sell_order(SYMBOL, final_contracts, params)
        
        log(f"✅ 주문 완료 ({final_contracts} contracts)")
    except Exception as e:
        log(f"❌ Order Fail: {e}")

def bot_loop():
    global is_running
    model, device = load_model()
    if not model: 
        log("No Model Found. Stopping.")
        stop_bot()
        return
        
    log("=== 🛡️ BOT STARTED ===")
    
    while is_running:
        try:
            size, _, pnl = get_my_position_info()
            
            # 1. 포지션 없을 때 (신규 진입)
            if size == 0:
                dec = get_1h_ai_decision(model, device)
                if dec == 'LONG' and check_entry_signal('LONG'): 
                    execute_order('buy')
                    time.sleep(10)
                elif dec == 'SHORT' and check_entry_signal('SHORT'): 
                    execute_order('sell')
                    time.sleep(10)
            
            # 2. 포지션 있을 때 (청산 관리 + 물타기)
            else:
                # ========================================================
                # [물타기 로직] 평단보다 가격이 안 좋을 때(손실 중)만 추가 진입
                # ========================================================
                if pnl < -0.5: # 0.5% 이상 손실 중일 때
                    dec = get_1h_ai_decision(model, device)
                    
                    # 롱 물타기
                    if size > 0 and dec == 'LONG' and check_entry_signal('LONG'):
                        log(f"💧 롱 물타기 진입! (수익률: {pnl:.2f}%)")
                        execute_order('buy')
                        time.sleep(10) 

                    # 숏 물타기
                    elif size < 0 and dec == 'SHORT' and check_entry_signal('SHORT'):
                        log(f"💧 숏 물타기 진입! (수익률: {pnl:.2f}%)")
                        execute_order('sell')
                        time.sleep(10)
                # ========================================================

                current_tp = float(entry_tp.get()) if entry_tp.get() else curr_tp_pct
                current_sl = float(entry_sl.get()) if entry_sl.get() else curr_sl_pct
                
                if pnl >= current_tp:
                    log(f"💰 Take Profit (+{pnl:.2f}%)")
                    execute_order('sell' if size > 0 else 'buy', reduce_only=True)
                    time.sleep(5)
                elif pnl <= -current_sl:
                    log(f"🏳️ Stop Loss (-{pnl:.2f}%)")
                    execute_order('sell' if size > 0 else 'buy', reduce_only=True)
                    time.sleep(5)
            
            # 루프 지연
            for _ in range(10): 
                if not is_running: break
                time.sleep(1)
                
        except Exception as e:
            log(f"Err: {e}")
            time.sleep(5)
    log("=== 🛑 BOT STOPPED ===")

def start_bot():
    global is_running, bot_thread, curr_leverage, curr_balance_pct, curr_tp_pct, curr_sl_pct
    if is_running: return
    try:
        curr_leverage = int(entry_leverage.get())
        curr_balance_pct = float(entry_balance.get())
        curr_tp_pct = float(entry_tp.get())
        curr_sl_pct = float(entry_sl.get())
        
        if curr_leverage < 1 or curr_leverage > 100:
            messagebox.showerror("Err", "레버리지: 1~100")
            return
        if curr_balance_pct < 1 or curr_balance_pct > 100:
            messagebox.showerror("Err", "투자비율: 1~100")
            return
            
    except ValueError:
        messagebox.showerror("Err", "숫자만 입력하세요")
        return
    
    log(f"시작 설정: 레버리지 {curr_leverage}x, 투자금 {curr_balance_pct}%")
    
    is_running = True
    btn_start.config(state='disabled', bg='#555555')
    btn_stop.config(state='normal', bg='#ff4444')
    entry_leverage.config(state='disabled'); entry_balance.config(state='disabled')
    entry_tp.config(state='disabled'); entry_sl.config(state='disabled')
    
    bot_thread = threading.Thread(target=bot_loop)
    bot_thread.daemon = True
    bot_thread.start()

def stop_bot():
    global is_running
    is_running = False
    log("Stopping...")
    time.sleep(1)
    btn_start.config(state='normal', bg='#00ff00')
    btn_stop.config(state='disabled', bg='#555555')
    entry_leverage.config(state='normal'); entry_balance.config(state='normal')
    entry_tp.config(state='normal'); entry_sl.config(state='normal')
    log("봇 정지됨.")

# ====================================================
# GUI 구성
# ====================================================
root = tk.Tk()
root.title("Gate.io Pro Sniper (Final Ver.)")
root.geometry("450x680")
root.configure(bg="#222222")

style = ttk.Style()
style.theme_use('clam')
style.configure("Dark.TLabelframe", background="#222222", foreground="white", bordercolor="#555555")
style.configure("Dark.TLabelframe.Label", background="#222222", foreground="#00ff00", font=("Arial", 12, "bold"))

# 계좌 현황 프레임
frame_dash = ttk.LabelFrame(root, text=" ACCOUNT ", style="Dark.TLabelframe")
frame_dash.pack(padx=10, pady=10, fill="x")
tk.Label(frame_dash, text="USDT:", bg="#222222", fg="#aaa").grid(row=0, column=0, sticky='w', padx=10)
lbl_balance_val = tk.Label(frame_dash, text="Loading...", bg="#222222", fg="white", font=("Arial", 12, "bold")); lbl_balance_val.grid(row=0, column=1, sticky='e')
tk.Label(frame_dash, text="BTC:", bg="#222222", fg="#aaa").grid(row=1, column=0, sticky='w', padx=10)
lbl_price_val = tk.Label(frame_dash, text="Loading...", bg="#222222", fg="#ff0", font=("Arial", 12, "bold")); lbl_price_val.grid(row=1, column=1, sticky='e')
tk.Label(frame_dash, text="POS:", bg="#222222", fg="#aaa").grid(row=2, column=0, sticky='w', padx=10)
lbl_pos_val = tk.Label(frame_dash, text="-", bg="#222222", fg="white", font=("Arial", 11)); lbl_pos_val.grid(row=2, column=1, sticky='e')
tk.Label(frame_dash, text="Entry:", bg="#222222", fg="#aaa").grid(row=3, column=0, sticky='w', padx=10)
lbl_entry_val = tk.Label(frame_dash, text="-", bg="#222222", fg="white", font=("Arial", 11)); lbl_entry_val.grid(row=3, column=1, sticky='e')

# 설정 프레임
frame_set = ttk.LabelFrame(root, text=" SETTINGS ", style="Dark.TLabelframe")
frame_set.pack(padx=10, pady=10, fill="x")
def mk_inp(r, c, t, d):
    tk.Label(frame_set, text=t, bg="#222222", fg="white").grid(row=r, column=c, padx=5, pady=5, sticky='e')
    e = tk.Entry(frame_set, width=8, bg="#333", fg="white", insertbackground="white"); e.insert(0, d); e.grid(row=r, column=c+1, padx=5, pady=5)
    return e
entry_leverage = mk_inp(0, 0, "Lev(x):", "10"); entry_balance = mk_inp(0, 2, "Size(%):", "50")
entry_tp = mk_inp(1, 0, "TP(%):", "10.0"); entry_sl = mk_inp(1, 2, "SL(%):", "5.0")

# 버튼
frame_btns = tk.Frame(root, bg="#222222"); frame_btns.pack(pady=10)
btn_start = tk.Button(frame_btns, text="▶ START", font=("Arial", 14, "bold"), bg="#0f0", width=10, command=start_bot); btn_start.pack(side=tk.LEFT, padx=10)
btn_stop = tk.Button(frame_btns, text="■ STOP", font=("Arial", 14, "bold"), bg="#555", fg="white", width=10, command=stop_bot, state='disabled'); btn_stop.pack(side=tk.LEFT, padx=10)

# 로그창
tk.Label(root, text="[ System Log ]", bg="#222222", fg="#aaa").pack(anchor='w', padx=10)
log_area = scrolledtext.ScrolledText(root, width=50, height=12, bg="#111", fg="#0f0", font=("Consolas", 9), state='disabled'); log_area.pack(padx=10, pady=5, fill='both', expand=True)

root.after(1000, update_dashboard)
root.mainloop()