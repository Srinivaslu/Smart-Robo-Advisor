
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings

# Ignore minor warnings to keep the terminal clean
warnings.filterwarnings('ignore')

print("🚀 Starting the AI Brain Update Pipeline...")

# ==========================================
# 1. FETCH MARKET DATA (200+ NSE STOCKS)
# ==========================================
print("📊 Downloading 3-year real historical data for 200+ stocks (This may take 30-60 seconds)...")

# Nifty 50 + Next 50 + Midcap 50 + Smallcap 50 + Core ETFs
ticker_symbols = [
    # --- LARGE CAP (Nifty 50) ---
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 
    'HINDUNILVR.NS', 'LT.NS', 'BAJFINANCE.NS', 'HCLTECH.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'TATAMOTORS.NS', 
    'TATASTEEL.NS', 'POWERGRID.NS', 'NTPC.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'M&M.NS', 'ASIANPAINT.NS', 
    'TITAN.NS', 'BAJAJFINSV.NS', 'ADANIENT.NS', 'ADANIPORTS.NS', 'NESTLEIND.NS', 'ONGC.NS', 'COALINDIA.NS', 
    'JSWSTEEL.NS', 'HAL.NS', 'HINDALCO.NS', 'WIPRO.NS', 'GRASIM.NS', 'TECHM.NS', 'DLF.NS', 'ZOMATO.NS', 
    'SBILIFE.NS', 'DRREDDY.NS', 'HDFCLIFE.NS', 'LTIM.NS', 'EICHERMOT.NS', 'APOLLOHOSP.NS', 'DIVISLAB.NS', 
    'CIPLA.NS', 'TATACONSUM.NS', 'BRITANNIA.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'INDUSINDBK.NS',
    
    # --- MID/SMALL CAP & NEXT 50 ---
    'TVSMOTOR.NS', 'TRENT.NS', 'CHOLAFIN.NS', 'BEL.NS', 'PIDILITIND.NS', 'SHRIRAMFIN.NS', 'AMBUJACEM.NS', 
    'SIEMENS.NS', 'CUMMINSIND.NS', 'BOSCHLTD.NS', 'PFC.NS', 'RECLTD.NS', 'GAIL.NS', 'INDIGO.NS', 'LODHA.NS', 
    'JINDALSTEL.NS', 'VEDL.NS', 'PNB.NS', 'BOB.NS', 'CANBK.NS', 'UNIONBANK.NS', 'IDFCFIRSTB.NS', 'IOB.NS', 
    'INDIANB.NS', 'TORNTPHARM.NS', 'AUROPHARMA.NS', 'LUPIN.NS', 'MANKIND.NS', 'MAXHEALTH.NS', 'ZYDUSLIFE.NS', 
    'ALKEM.NS', 'GODREJCP.NS', 'DABUR.NS', 'COLPAL.NS', 'MARICO.NS', 'UBL.NS', 'UNITEDSPR.NS', 'DMART.NS', 
    'PGHH.NS', 'AWL.NS', 'PATANJALI.NS', 'PAGEIND.NS', 'SRF.NS', 'PIIND.NS', 'DEEPAKNTR.NS', 'NAVINFLUOR.NS', 
    'AARTIIND.NS', 'TATACHEM.NS', 'COROMANDEL.NS', 'FLUOROCHEM.NS', 'VOLTAS.NS', 'HAVELLS.NS', 'CROMPTON.NS', 
    'DIXON.NS', 'POLYCAB.NS', 'KEI.NS', 'ABFRL.NS', 'BATAINDIA.NS', 'RELAXO.NS', 'ASTRAL.NS', 'SUPREMEIND.NS', 
    'FINCABLES.NS', 'MRF.NS', 'APOLLOTYRE.NS', 'BALKRISIND.NS', 'ESCORTS.NS', 'ASHOKLEY.NS', 'MOTHERSON.NS', 
    'SONACOMS.NS', 'MINDAIND.NS', 'CONCOR.NS', 'IRCTC.NS', 'NYKAA.NS', 'PAYTM.NS', 'POLICYBZR.NS', 'DELHIVERY.NS',
    'CARTRADE.NS', 'ZENSARTECH.NS', 'CYIENT.NS', 'KPITTECH.NS', 'TATAELXSI.NS', 'PERSISTENT.NS', 'COFORGE.NS', 
    'MPHASIS.NS', 'OFSS.NS', 'LTTS.NS', 'L&TFH.NS', 'M&MFIN.NS', 'MUTHOOTFIN.NS', 'MANAPPURAM.NS', 'SUNDARMFIN.NS', 
    'LICHSGFIN.NS', 'PNBHOUSING.NS', 'CANFINHOME.NS', 'AUBANK.NS', 'BANDHANBNK.NS', 'FEDERALBNK.NS', 'CUB.NS', 
    'KARURVYSYA.NS', 'RBLBANK.NS', 'SYNGENE.NS', 'BIOCON.NS', 'LAURUSLABS.NS', 'GRANULES.NS', 'IPCALAB.NS', 
    'NATCOPHARM.NS', 'ABBOTINDIA.NS', 'GLAXO.NS', 'PFIZER.NS', 'SANOFI.NS', 'APLLTD.NS', 'GUJGASLTD.NS', 'IGL.NS', 
    'MGL.NS', 'ATGL.NS', 'PETRONET.NS', 'OIL.NS', 'HINDPETRO.NS', 'BPCL.NS', 'IOC.NS', 'CASTROLIND.NS', 'SUZLON.NS', 
    'IREDA.NS', 'SJVN.NS', 'NHPC.NS', 'TORNTPOWER.NS', 'CESC.NS', 'IEX.NS', 'MCX.NS', 'BSE.NS', 'CDSL.NS', 
    'CAMS.NS', 'KFINTECH.NS', 'ANGELONE.NS', 'MOTILALOFS.NS', 'NAM-INDIA.NS', 'HDFCAMC.NS', 'UTIAMC.NS', 'RADICO.NS', 
    'BALRAMCHIN.NS', 'TRIDENT.NS', 'WELSPUNIND.NS', 'RAYMOND.NS', 'KALYANKJIL.NS', 'DEVYANI.NS', 'JUBIQUANT.NS', 
    'WESTLIFE.NS', 'SAPPHIRE.NS', 'PVRINOX.NS', 'SUNTV.NS', 'ZEEL.NS', 'NETWORK18.NS', 'TV18BRDCST.NS', 'IRB.NS', 
    'KNRCON.NS', 'PNCINFRA.NS', 'NCC.NS', 'NBCC.NS', 'ENGINERSIN.NS', 'RITES.NS', 'RVNL.NS', 'IRCON.NS', 
    'MAZDOCK.NS', 'COCHINSHIP.NS', 'GRSE.NS', 'BDL.NS', 'BEML.NS', 'MTARTECH.NS', 'DATAATTNS.NS',
    
    # --- CORE ETFs (Required for Safe portfolios) ---
    'NIFTYBEES.NS', 'GOLDBEES.NS', 'LIQUIDBEES.NS', 'JUNIORBEES.NS'
]

# yfinance will download all of them in bulk!
historical_data = yf.download(ticker_symbols, period="3y", interval="1d", progress=False)['Close']

portfolio_data = []
print("🧮 Calculating real Annual Returns and Volatility for all stocks...")

for ticker in ticker_symbols:
    try:
        stock_prices = historical_data[ticker].dropna()
        if len(stock_prices) < 200:
            continue # Skip if the stock IPO'd very recently and lacks data
            
        # 1. CAGR (Annual Return)
        start_price = stock_prices.iloc[0]
        end_price = stock_prices.iloc[-1]
        years = len(stock_prices) / 252 
        cagr = ((end_price / start_price) ** (1 / years) - 1) * 100
        
        # 2. Volatility (Annualized)
        daily_returns = stock_prices.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100 
        
        clean_symbol = ticker.replace('.NS', '')
        
        portfolio_data.append({
            'Stock_Name': clean_symbol, 
            'Symbol': clean_symbol,
            'Sector': 'Equity', # A general tag
            'Annual_Return': round(cagr, 2),
            'Volatility_Beta': round(volatility / 100, 2)
        })
    except Exception:
        pass # Silently ignore symbols that fail so it doesn't crash the script

df = pd.DataFrame(portfolio_data)
print(f"✅ Successfully processed {len(df)} live stocks!")

# ==========================================
# 2. RUN K-MEANS CLUSTERING
# ==========================================
print("🧠 Grouping massive dataset into Risk Clusters using K-Means...")

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster_ID'] = kmeans.fit_predict(df[['Annual_Return', 'Volatility_Beta']])

cluster_means = df.groupby('Cluster_ID')['Volatility_Beta'].mean().sort_values()
safe_id = cluster_means.index[0]
moderate_id = cluster_means.index[1]
risky_id = cluster_means.index[2]

def assign_risk(cluster_id):
    if cluster_id == safe_id: return "Safe"
    elif cluster_id == moderate_id: return "Moderate"
    else: return "Risky"

df['Risk_Category'] = df['Cluster_ID'].apply(assign_risk)

df.to_csv('clustered_stocks.csv', index=False)
print("✅ Updated 'clustered_stocks.csv' successfully.")

# ==========================================
# 3. RE-TRAIN THE RANDOM FOREST MODEL
# ==========================================
print("🌳 Retraining the User Risk Profiling Model...")

ages = np.random.randint(18, 75, 500)
incomes = np.random.randint(20000, 500000, 500)
scores = np.random.randint(6, 31, 500)
X_train = pd.DataFrame({'Age': ages, 'Income': incomes, 'Risk_Score': scores})

def determine_profile(row):
    if row['Risk_Score'] >= 24 and row['Age'] < 40: return "Aggressive"
    elif row['Risk_Score'] <= 12 or row['Age'] > 60: return "Conservative"
    else: return "Moderate"

y_train = X_train.apply(determine_profile, axis=1)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train.values, y_train)

joblib.dump(rf_model, 'risk_model.pkl')
print("✅ Updated 'risk_model.pkl' successfully.")

print("🎉 Pipeline Complete! Your app now runs on a 200+ stock universe.")