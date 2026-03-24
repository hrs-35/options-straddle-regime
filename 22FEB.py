import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import time

# ================= CONFIGURATION =================
FILE_5MIN = r"C:\Users\user\Desktop\test1\FINAL_NIFTY_MASTER_ATM_5min.csv"
FILE_1MIN = r"C:\Users\user\Desktop\test1\FINAL_NIFTY_MASTER_ATM.csv"

STARTING_EQUITY = 500000
RISK_PER_TRADE = 0.01          
STOP_LOSS_PCT = 0.15            
MARGIN_PER_LOT = 100000        
LOT_UNIT = 65
SLIPPAGE_PER_SIDE = 0.0005     

# --- TAX & CHARGES ---
BROKERAGE_PER_ORDER = 20.0     
STT_SELL_PCT = 0.001           
EXCHANGE_TX_PCT = 0.0005       
GST_PCT = 0.18                 

# Filters
TREND_THRESHOLD = 0.001
REGIME_IV_GREEN = 25
REGIME_IV_YELLOW = 55      
DAILY_LOSS_LIMIT = -0.01

# ================= DATA ENGINE =================
def load_production_data():
    print("Scrubbing and Calibrating Data...")
    df1 = pd.read_csv(FILE_1MIN)
    df5 = pd.read_csv(FILE_5MIN)
    
    processed_dfs = []
    for df in [df1, df5]:
        df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
        df.sort_values('datetime', inplace=True)
        df['straddle'] = df['CE_close'] + df['PE_close']
        # Global filter to remove pre-market/auction data
        df = df[df['datetime'].dt.time > time(9,15)].copy()
        df = df[(df['iv'] < 100) & (df['iv'] > 2)].copy()
        processed_dfs.append(df)
        
    df1, df5 = processed_dfs[0], processed_dfs[1]
    df5['IV_Rank'] = (df5['iv'] - df5['iv'].rolling(100).min()) / (df5['iv'].rolling(100).max() - df5['iv'].rolling(100).min() + 1e-9) * 100
    df5['Trend'] = df5['spot'].pct_change(10).abs()
    df5['Compression'] = df5['straddle'].rolling(10).std() / df5['straddle'].rolling(10).mean()
    
    def define_regime(row):
        if row['IV_Rank'] > REGIME_IV_YELLOW or row['Trend'] > TREND_THRESHOLD: return 'RED'
        if row['IV_Rank'] > REGIME_IV_GREEN: return 'YELLOW'
        return 'GREEN'
    
    df5['Regime'] = df5.apply(define_regime, axis=1)
    
    # MODIFIED: Ensure signals are only mapped within the 9:45 AM - 3:00 PM window
    signal_map = {}
    valid_signals = df5[
        (df5['Regime'] != 'RED') & 
        (df5['Compression'] < 0.028) &
        (df5['datetime'].dt.time >= time(9, 45)) & 
        (df5['datetime'].dt.time <= time(15, 0))
    ]
    for _, row in valid_signals.iterrows():
        signal_map[row['datetime']] = row['Regime']
            
    return df1, signal_map

# ================= ANALYTICS & VISUALIZATION =================
class PortfolioVisualizer:
    def __init__(self, df):
        self.df = df
        self.df['Entry_Date'] = pd.to_datetime(self.df['Entry_Date'])
        
    def plot_full_dashboard(self):
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(20, 12))
        fig.patch.set_facecolor('#0E1117') 
        
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
        ax1.plot(self.df['Entry_Date'], self.df['Equity'], color='#00FF41', linewidth=2)
        ax1.set_title("Compounding Equity Curve", fontsize=14, color='white')
        ax1.fill_between(self.df['Entry_Date'], STARTING_EQUITY, self.df['Equity'], color='#00FF41', alpha=0.1)
        ax1.grid(True, alpha=0.1)

        ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
        ax2.fill_between(self.df['Entry_Date'], 0, -self.df['Drawdown_Pct'], color='#FF3131', alpha=0.5)
        ax2.set_title("Drawdown Percentage", fontsize=12, color='white')
        ax2.grid(True, alpha=0.1)

        ax3 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
        monthly_pnl = self.df.copy()
        monthly_pnl['Year'] = monthly_pnl['Entry_Date'].dt.year
        monthly_pnl['Month'] = monthly_pnl['Entry_Date'].dt.month
        pivot_table = monthly_pnl.pivot_table(index='Year', columns='Month', values='Net_PnL', aggfunc='sum').fillna(0)
        sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="RdYlGn", ax=ax3, cbar=False, center=0)
        ax3.set_title("Monthly PnL Heatmap", color='white')

        ax4 = plt.subplot2grid((3, 3), (2, 0))
        sns.histplot(self.df['Net_PnL'], kde=True, ax=ax4, color='#00D4FF')
        ax4.set_title("Return Distribution", color='white')

        ax5 = plt.subplot2grid((3, 3), (2, 1))
        self.df['Reason'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax5, colors=sns.color_palette("magma"))
        ax5.set_title("Exit Reasons", color='white')
        ax5.set_ylabel('')

        ax6 = plt.subplot2grid((3, 3), (2, 2))
        self.df.groupby('Regime')['Net_PnL'].sum().plot(kind='bar', ax=ax6, color=['#00FF41', '#FFBD03'])
        ax6.set_title("Profit by Regime", color='white')
        plt.xticks(rotation=0)

        plt.tight_layout()
        plt.show()

def calculate_trade_costs(entry_price, exit_price, lots):
    total_qty = lots * LOT_UNIT
    turnover = (entry_price + exit_price) * total_qty
    brokerage = BROKERAGE_PER_ORDER * 2 
    stt = (exit_price * total_qty) * STT_SELL_PCT
    tx_charges = turnover * EXCHANGE_TX_PCT
    gst = (brokerage + tx_charges) * GST_PCT
    return brokerage + stt + tx_charges + gst

# ================= BACKTEST ENGINE =================
def run_production_backtest(df1, signal_map):
    trades = []
    equity = STARTING_EQUITY
    in_trade = False
    current_date = None
    daily_pnl = 0

    for row in df1.itertuples():
        dt = row.datetime
        curr_time = dt.time()

        if dt.date() != current_date:
            daily_pnl = 0
            current_date = dt.date()

        if daily_pnl / equity < DAILY_LOSS_LIMIT: continue

        # MODIFIED: Only enter if current time is within 9:45 AM and 3:00 PM
        if not in_trade and dt in signal_map:
            if time(9, 45) <= curr_time <= time(15, 0):
                regime = signal_map[dt]
                # Entry calculations with slippage
                ce_entry_price = row.CE_close
                pe_entry_price = row.PE_close
                entry_straddle = (ce_entry_price + pe_entry_price) * (1 + SLIPPAGE_PER_SIDE)
                
                risk_amt = equity * RISK_PER_TRADE
                max_lots = min(int(risk_amt / (entry_straddle * STOP_LOSS_PCT * LOT_UNIT)), int(equity / MARGIN_PER_LOT))
                current_lots = max_lots if regime == 'GREEN' else max(1, int(max_lots * 0.5))
                
                if current_lots < 1: continue
                
                in_trade, entry_time, entry_regime = True, dt, regime
                ce_sold, pe_sold, realised_pts, bar_count = False, False, 0, 0
                ce_exit_price, pe_exit_price = 0, 0
                continue

        if in_trade:
            bar_count += 1
            curr_ce, curr_pe = row.CE_close, row.PE_close
            
            # Leg-wise Stop Loss
            if not ce_sold and (curr_ce / ce_entry_price - 1) >= 0.25:
                ce_sold = True
                ce_exit_price = curr_ce * (1 - SLIPPAGE_PER_SIDE)
                realised_pts += ce_exit_price
            if not pe_sold and (curr_pe / pe_entry_price - 1) >= 0.25:
                pe_sold = True
                pe_exit_price = curr_pe * (1 - SLIPPAGE_PER_SIDE)
                realised_pts += pe_exit_price

            total_val = (0 if ce_sold else curr_ce) + (0 if pe_sold else curr_pe) + realised_pts
            pnl_pct = (total_val - entry_straddle) / entry_straddle

            reason = None
            if pnl_pct >= 0.12: reason = "TARGET"
            elif pnl_pct <= -STOP_LOSS_PCT: reason = "STOP_LOSS"
            elif bar_count >= 30: reason = "TIME_OUT"
            elif curr_time >= time(15, 10): reason = "EOD"

            if reason:
                if not ce_sold: ce_exit_price = curr_ce * (1 - SLIPPAGE_PER_SIDE)
                if not pe_sold: pe_exit_price = curr_pe * (1 - SLIPPAGE_PER_SIDE)
                
                exit_val = (ce_exit_price + pe_exit_price)
                gross = (exit_val - entry_straddle) * (current_lots * LOT_UNIT)
                tax = calculate_trade_costs(entry_straddle, exit_val, current_lots)
                net = gross - tax
                
                total_slippage = (entry_straddle * SLIPPAGE_PER_SIDE + exit_val * SLIPPAGE_PER_SIDE) * (current_lots * LOT_UNIT)
                
                equity += net
                daily_pnl += net
                trades.append({
                    'Entry_Date': entry_time,
                    'Exit_Date': dt,
                    'Regime': entry_regime,
                    'Lots': current_lots,
                    'CE_Entry': round(ce_entry_price, 2),
                    'PE_Entry': round(pe_entry_price, 2),
                    'Straddle_Entry': round(entry_straddle, 2),
                    'CE_Exit': round(ce_exit_price, 2),
                    'PE_Exit': round(pe_exit_price, 2),
                    'Straddle_Exit': round(exit_val, 2),
                    'Gross_PnL': round(gross, 2),
                    'Taxes': round(tax, 2),
                    'Slippage_Cost': round(total_slippage, 2),
                    'Net_PnL': round(net, 2),
                    'Equity': round(equity, 2),
                    'Reason': reason
                })
                in_trade = False
    
    res = pd.DataFrame(trades)
    if not res.empty:
        res['Peak'] = res['Equity'].cummax()
        res['Drawdown_Pct'] = round((res['Peak'] - res['Equity']) / res['Peak'] * 100, 2)
    return res

# ================= EXECUTION =================
try:
    df1_data, sig_map = load_production_data()
    results = run_production_backtest(df1_data, sig_map)
    
    if not results.empty:
        wins = results[results['Net_PnL'] > 0]['Net_PnL']
        losses = results[results['Net_PnL'] < 0]['Net_PnL']
        
        print("\n" + "═"*65)
        print("                DETAILED STRATEGY ANALYSIS")
        print("═"*65)
        metrics = {
            "Total Trades": len(results),
            "Win Rate": f"{(len(wins)/len(results))*100:.2f}%",
            "Profit Factor": f"{wins.sum() / abs(losses.sum()):.2f}" if not losses.empty else "Inf",
            "Max Drawdown": f"{results['Drawdown_Pct'].max():.2f}%",
            "Total Net Profit": f"₹{results['Net_PnL'].sum():,.2f}",
            "Avg Profit/Trade": f"₹{results['Net_PnL'].mean():.2f}",
            "Total Taxes Paid": f"₹{results['Taxes'].sum():,.2f}",
            "Est. Slippage Paid": f"₹{results['Slippage_Cost'].sum():,.2f}"
        }
        for k, v in metrics.items(): print(f"{k:<30}: {v}")

        print("\n" + "─"*65)
        print("TRADE LOG SNIPPET (Leg-wise Pricing)")
        print("─"*65)
        disp_cols = ['Entry_Date', 'Regime', 'CE_Entry', 'PE_Entry', 'CE_Exit', 'PE_Exit', 'Net_PnL', 'Reason']
        print(pd.concat([results.head(3), results.tail(3)])[disp_cols])
        
        results.to_csv("Enhanced_TradeLog_with_Legs.csv", index=False)
        print(f"\n[Success] Full audit log saved: Enhanced_TradeLog_with_Legs.csv")
        
        viz = PortfolioVisualizer(results)
        viz.plot_full_dashboard()
    else:
        print("\n[!] No trades within the 9:45-15:00 window. Check data/filters.")

except Exception as e:
    print(f"\n[Critical Error]: {e}")
