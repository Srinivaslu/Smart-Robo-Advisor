import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
import datetime  
from openai import OpenAI
import gspread
from google.oauth2.service_account import Credentials
import json

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Smart Robo-Advisor", layout="wide")

# ==========================================
# 2. LOAD TRAINED MODELS & DATA
# ==========================================
# We wrap this in a try-except block to handle missing files gracefully
try:
    risk_model = joblib.load('risk_model.pkl')
    stock_df = pd.read_csv('clustered_stocks.csv')
except FileNotFoundError:
    st.error("🚨 CRITICAL ERROR: Model files not found!")
    st.warning("Please run 'training_logics.ipynb' first to generate 'risk_model.pkl' and 'clustered_stocks.csv'.")
    st.stop()
    
    

# ==========================================
# 3. HELPER FUNCTIONS (MATH, LOGIC & FORMATTING)
# ==========================================

import json # Make sure this is at the top of your file!

def save_to_google_sheets(data_list):
    """Silently pushes a row of user data to the Google Sheet."""
    try:
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        
        # --- NEW: CLOUD VS LOCAL LOGIC ---
        if "gcp_credentials" in st.secrets:
            # We are on Streamlit Cloud! Read the secret password securely.
            creds_dict = json.loads(st.secrets["gcp_credentials"])
            creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        else:
            # We are testing locally on your laptop. Use the file.
            creds = Credentials.from_service_account_file('google_credentials.json', scopes=scopes)
        
        client = gspread.authorize(creds)
        
        # Make sure this matches the exact name of your Google Sheet!
        sheet = client.open('MBA_Robo_Advisor_Data').sheet1 
        sheet.append_row(data_list)
        
    except Exception as e:
        print(f"⚠️ Could not save to Google Sheets: {e}")
        
        # We use a print statement instead of st.error so it fails silently 
        # and doesn't interrupt the user's experience if the internet blips!

def format_indian(number):
    """
    Converts a number to Indian Lakh/Crore format string.
    Example: 100000 -> ₹1.0 L, 10000000 -> ₹1.0 Cr
    """
    number = int(number)
    if number >= 10000000:
        return f"₹{number / 10000000:.2f} Cr"
    elif number >= 100000:
        return f"₹{number / 100000:.2f} L"
    else:
        # Standard comma formatting for smaller numbers
        return f"₹{number:,}"

def calculate_surplus(income, expenses, emi):
    """Calculates the monthly money available for investment."""
    return income - expenses - emi

def check_debt_status(loan_rate):
    """Checks if the user's debt interest is too high to invest safely."""
    if loan_rate > 12.0:
        return "PRIORITY: PAY DEBT"
    return "SAFE"

import yfinance as yf

@st.cache_data(ttl=3600) # Cache data for 1 hour to prevent slowness
def get_live_prices(ticker_list):
    """
    Fetches live prices for Indian stocks from Yahoo Finance.
    Append '.NS' for NSE stocks.
    """
    live_data = {}
    try:
        # We assume tickers in your CSV don't have '.NS', so we add it
        # Note: In a real app, your CSV should have a 'Symbol' column like 'RELIANCE'
        # For this demo, we will just fetch a benchmark like Nifty 50 if specific symbols aren't in CSV
        # BUG FIX: Only add '.NS' if the ticker doesn't already have a '.' (like .BO)
        tickers = [t if t.startswith('^') or '.' in t else t + ".NS" for t in ticker_list]
        data = yf.download(tickers, period="1d", progress=False)['Close']
        
        # If single stock, data is a Series; if multiple, it's a DataFrame
        if not data.empty:
            current_vals = data.iloc[-1] # Get latest closing price
            live_data = current_vals.to_dict()
    except Exception as e:
        print(f"API Error: {e}")
        return None
        
    return live_data

# def generate_smart_strategy(user_profile, age, wellness_score, debt_status, surplus):
#     """Generates personalized text advice based on user data."""
#     tips = []
    
#     # 1. AGE BASED ADVICE
#     if age < 30:
#         tips.append("🌱 **Age Advantage:** You are in the 'Golden Accumulation Phase'. Your biggest asset is Time. Even small SIPs now will compound massively over 20+ years.")
#     elif age < 50:
#         tips.append("⚓ **Peak Earning Phase:** Focus on increasing your SIP step-up rate. This is the time to maximize tax-saving instruments (NPS/PPF) alongside equity.")
#     else:
#         tips.append("🛡️ **Preservation Phase:** As you approach retirement, shift focus from 'High Growth' to 'Stable Income'. Reduce exposure to small-cap stocks.")

#     # 2. SCORE BASED ADVICE
#     if wellness_score < 50:
#         tips.append("🚨 **Financial First Aid:** Your Wellness Score is low. Before aggressive investing, build your Emergency Fund (6 months expenses) and clear loans > 10% interest.")
#     elif wellness_score > 80:
#         tips.append("🚀 **Wealth Accelerator:** Your basics are strong. Consider diversifying into 'Alternative Assets' like REITs or International Funds for the next leg of growth.")

#     # 3. PROFILE SPECIFIC
#     if user_profile == "Conservative":
#         tips.append("⚠️ **Inflation Alert:** Being too safe is also risky. Fixed Deposits often lose value against inflation. Stick to the recommended 20% Equity allocation to beat inflation.")
#     elif user_profile == "Aggressive":
#         tips.append("📉 **Volatility Warning:** Your portfolio is high-growth but volatile. Do not panic sell if the market drops 10-20%. View corrections as buying opportunities.")

#     return tips

def generate_smart_strategy(user_profile, age, wellness_score, debt_status, surplus):
    """Generates personalized text advice using local Phi-3 via Ollama."""
    
    try:
        # Connect to your local Ollama server
        client = OpenAI(
            base_url='http://localhost:11434/v1/',
            api_key='ollama', # Dummy key required by the library
        )
        
        # Build the prompt with the user's actual data
        prompt = f"""
        You are an expert Indian financial advisor. Generate 3 short, personalized financial tips for a client with this profile:
        - Age: {age}
        - Risk Profile: {user_profile}
        - Financial Wellness Score (0-100): {wellness_score}
        - Debt Status: {debt_status}
        - Monthly Investment Surplus: ₹{surplus}

        Instructions: Provide exactly 3 tips. Format as a bulleted list using '-'. Keep each tip to one sentence. Start each with an emoji.
        """
        
        # Ask Phi-3 to generate the tips
        response = client.chat.completions.create(
            model='phi3',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.7
        )
        
        # Clean up the text so it fits nicely in your Streamlit app and PDF
        raw_text = response.choices[0].message.content.strip()
        tips = [tip.strip('-* ') for tip in raw_text.split('\n') if tip.strip()]
        
        return tips

    except Exception as e:
        print(f"Local AI Error: {e}")
        return ["⚠️ **AI Offline:** Make sure Ollama is running in the background."]

# --- REPLACE YOUR EXISTING WEALTH & PROJECTION FUNCTIONS WITH THESE TWO ---

def calculate_future_wealth_dynamic(income, expenses, loan_df, rate_return, years, step_up_rate, inflation_rate=0.06):
    """
    Advanced Wealth Calculation:
    1. Monthly Step-Up (Salary Hike)
    2. Dynamic EMI Freedom (Invest EMI amount after loan ends)
    """
    if years <= 0: return 0, 0, 0, 0
    
    # --- CRASH FIX: CLEAN DATA BEFORE MATH ---
    # Convert any missing/empty values (None) to 0.0 to prevent crash
    if not loan_df.empty:
        loan_df = loan_df.fillna(0)
    
    monthly_rate = (rate_return / 100) / 12
    months = years * 12
    
    # 1. Calculate Initial Total EMI
    current_total_emi = 0
    if not loan_df.empty:
        current_total_emi = loan_df['EMI Amount (₹)'].sum()
    
    # 2. Build a "Freedom Schedule" (When does each loan end?)
    loan_schedule = np.zeros(months + 1)
    
    if not loan_df.empty:
        for _, row in loan_df.iterrows():
            emi_amt = row['EMI Amount (₹)']
            # Safety check: Ensure years is a number
            years_rem = float(row['Years Remaining']) if row['Years Remaining'] is not None else 0.0
            
            months_left = int(years_rem * 12)
            
            if months_left < months:
                loan_schedule[months_left] += emi_amt
    
    # Initial Monthly Investment
    current_investment = income - expenses - current_total_emi
    
    current_corpus = 0
    total_invested = 0
    
    # 3. Month-by-Month Simulation
    for m in range(1, months + 1):
        # A. Check if any loan ended last month?
        freed_emi = loan_schedule[m-1] 
        if freed_emi > 0:
            current_investment += freed_emi
            
        # B. Invest and Grow
        current_corpus += current_investment
        current_corpus *= (1 + monthly_rate)
        total_invested += current_investment
        
        # C. Yearly Step-Up (Salary Hike)
        if m % 12 == 0:
            current_investment *= (1 + step_up_rate/100)
            
    future_value = int(current_corpus)
    real_value = int(future_value / ((1 + inflation_rate) ** years))
    
    return future_value, real_value, int(total_invested), current_total_emi

def get_projection_data_dynamic(income, expenses, loan_df, rate_return, years, step_up_rate):
    """Generates Chart Data considering Loan End Dates."""
    
    # --- CRASH FIX: CLEAN DATA FIRST ---
    if not loan_df.empty:
        loan_df = loan_df.fillna(0) # Convert "None" or Empty to 0
    
    data = []
    total_invested = 0
    current_corpus = 0
    monthly_rate = (rate_return / 100) / 12
    months = years * 12
    
    # Re-calculate Initial EMI & Schedule (Same logic as above)
    current_total_emi = 0
    loan_schedule = np.zeros(months + 1)
    
    if not loan_df.empty:
        current_total_emi = loan_df['EMI Amount (₹)'].sum()
        for _, row in loan_df.iterrows():
            months_left = int(row['Years Remaining'] * 12)
            if months_left < months:
                loan_schedule[months_left] += row['EMI Amount (₹)']

    current_investment = income - expenses - current_total_emi
    
    for m in range(1, months + 1):
        # Add freed EMI
        if loan_schedule[m-1] > 0:
            current_investment += loan_schedule[m-1]
            
        current_corpus += current_investment
        current_corpus *= (1 + monthly_rate)
        total_invested += current_investment
        
        if m % 12 == 0:
            current_investment *= (1 + step_up_rate/100)
            # Record Year End Data
            data.append({
                "Year": int(m/12),
                "Principal Invested": int(total_invested),
                "Wealth Created (Interest)": int(current_corpus - total_invested)
            })
    return pd.DataFrame(data).set_index("Year")
  
def check_goal_feasibility(surplus, goal_cost, years, expected_return):
    """Checks if the projected wealth meets the specific goal cost."""
    months = years * 12
    monthly_rate = (expected_return / 100) / 12
    # Simple Future Value formula for Goal Check
    future_value = surplus * ((((1 + monthly_rate) ** months) - 1) / monthly_rate) * (1 + monthly_rate)
    
    return future_value >= goal_cost, int(future_value)

def calculate_wellness_score(income, expenses, total_emi, emergency_fund_status, has_health, has_term, investment_list):
    """
    Advanced Logic: 
    - Savings (30%) + Debt (20%) + Emergency (10%) 
    - Insurance (20%) + Investment Diversity (20%)
    """
    score = 0
    
    # 1. Savings Rate (Max 30 points)
    monthly_surplus = income - expenses - total_emi
    if income > 0:
        savings_ratio = (monthly_surplus / income) * 100
    else:
        savings_ratio = 0
        
    if savings_ratio >= 30: score += 30
    elif savings_ratio >= 20: score += 20
    elif savings_ratio >= 10: score += 10
    
    # 2. Debt Burden (Max 20 points)
    if income > 0:
        dti = (total_emi / income) * 100
    else:
        dti = 0
        
    if dti == 0: score += 20
    elif dti < 30: score += 15
    elif dti < 40: score += 5
    # > 40% debt gets 0 points
    
    # 3. Emergency Fund Potential (Max 10 points)
    if emergency_fund_status == "Safe": score += 10
    else: score += 5
    
    # 4. Insurance Protection (Max 20 points) - CRITICAL FOR SAFETY
    if has_health: score += 10
    if has_term: score += 10
    
    # 5. Investment Habits (Max 20 points) - REWARDS DIVERSITY
    # We give points based on how many DIFFERENT assets they own
    unique_investments = len(investment_list)
    
    if unique_investments >= 3: score += 20  # Excellent Diversification
    elif unique_investments == 2: score += 15 # Good Start
    elif unique_investments == 1: score += 10 # Basic
    
    return min(100, score) # Cap at 100

def get_diversified_stocks(risk_category, need_tax_saving):
    # 1. BUFFETT RULE (Safe = ETFs)
    if risk_category == "Safe":
        data = [
            {'Stock_Name': 'Nifty 50 ETF', 'Symbol': 'NIFTYBEES', 'Sector': 'Index Fund', 'Annual_Return': 12.5, 'Volatility_Beta': 1.0},
            {'Stock_Name': 'Gold BeES', 'Symbol': 'GOLDBEES', 'Sector': 'Commodity', 'Annual_Return': 8.5, 'Volatility_Beta': 0.25},
            {'Stock_Name': 'Liquid BeES', 'Symbol': 'LIQUIDBEES', 'Sector': 'Debt', 'Annual_Return': 6.0, 'Volatility_Beta': 0.05},
            {'Stock_Name': 'Nifty Next 50', 'Symbol': 'JUNIORBEES', 'Sector': 'Midcap Index', 'Annual_Return': 14.0, 'Volatility_Beta': 1.1},
        ]
        df = pd.DataFrame(data)
    else:
        # 2. SPECIALIST RULE (Sharpe Ratio)
        df = stock_df[stock_df['Risk_Category'] == risk_category].copy()
        df['Sharpe_Ratio'] = df['Annual_Return'] / (df['Volatility_Beta'] + 0.01)
        df = df.sort_values(by='Sharpe_Ratio', ascending=False)
        
        portfolio = []
        seen_sectors = set()
        for _, row in df.iterrows():
            if row['Sector'] not in seen_sectors:
                portfolio.append(row)
                seen_sectors.add(row['Sector'])
            if len(portfolio) >= 5: break
        
        df = pd.DataFrame(portfolio)
        # --- We now pull the Symbol column too ---
        df = df[['Stock_Name', 'Symbol', 'Sector', 'Annual_Return', 'Volatility_Beta']]

    # 3. RACHANA RANADE RULE (Tax Saving)
    if need_tax_saving:
        elss_fund = pd.DataFrame([{
            'Stock_Name': 'Quant Tax Plan', 
            'Symbol': '0P0000XW8F.BO', # BSE mutual fund ticker
            'Sector': 'Tax Saving MF', 
            'Annual_Return': 15.5, 
            'Volatility_Beta': 1.1
        }])
        df = pd.concat([elss_fund, df], ignore_index=True)
        df = df.head(6)

    return df

# ==========================================
# 4. PDF GENERATOR (Now with Strategy & Lifestyle Check)
# ==========================================
def create_pdf_report(user_profile, monthly_surplus, future_val, allocation, rec_stocks, emergency_fund, insurance_needed, total_emi, wellness_score, strategic_tips, monthly_passive_income, future_expenses):
    def safe_text(text): return str(text).replace("₹", "Rs. ")

    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica', 'B', 16)
            self.cell(0, 10, 'AI Robo-Advisor Financial Plan', 0, 1, 'C')
            self.set_font('Helvetica', 'I', 10)
            self.cell(0, 10, f'Generated on: {datetime.date.today()}', 0, 1, 'C')
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    
    # --- 1. EXECUTIVE SUMMARY ---
    pdf.set_font("Helvetica", 'B', 14); pdf.cell(200, 10, txt="1. Executive Summary", ln=True)
    pdf.set_font("Helvetica", size=12)
    pdf.cell(200, 10, txt=f"Investor Profile: {user_profile}", ln=True)
    pdf.cell(200, 10, txt=f"Financial Wellness Score: {wellness_score}/100", ln=True)
    pdf.ln(2)
    pdf.cell(200, 10, txt=f"Current Monthly Surplus: {safe_text(format_indian(monthly_surplus))}", ln=True)
    pdf.cell(200, 10, txt=f"Current Total EMI: {safe_text(format_indian(total_emi))}", ln=True)
    pdf.cell(200, 10, txt=f"Projected Retirement Corpus: {safe_text(format_indian(future_val))}", ln=True)
    pdf.ln(5)

    # --- 2. RETIREMENT LIFESTYLE CHECK (New) ---
    pdf.set_font("Helvetica", 'B', 14); pdf.cell(200, 10, txt="2. Retirement Lifestyle Check", ln=True)
    pdf.set_font("Helvetica", size=12)
    pdf.cell(200, 10, txt=f"Projected Monthly Passive Income: {safe_text(format_indian(int(monthly_passive_income)))}", ln=True)
    pdf.cell(200, 10, txt=f"Est. Monthly Expenses (Inflation Adj): {safe_text(format_indian(int(future_expenses)))}", ln=True)
    
    if monthly_passive_income >= future_expenses:
        pdf.set_text_color(0, 128, 0) # Green
        pdf.cell(200, 10, txt="Status: COMFORTABLE (Passive Income > Expenses)", ln=True)
    else:
        pdf.set_text_color(255, 0, 0) # Red
        pdf.cell(200, 10, txt="Status: SHORTFALL DETECTED (Increase SIP Recommended)", ln=True)
    
    pdf.set_text_color(0, 0, 0) # Reset to Black
    pdf.ln(5)
    
    # --- 3. RECOMMENDED ALLOCATION ---
    pdf.set_font("Helvetica", 'B', 14); pdf.cell(200, 10, txt="3. Recommended Asset Allocation", ln=True)
    pdf.set_font("Helvetica", size=12)
    
    # NEW: Updated to match 'Debt Funds'
    pdf.cell(200, 10, txt=f"Equity: {allocation['Equity']}% | Debt Funds: {allocation['Debt Funds']}% | Gold: {allocation['Gold']}%", ln=True)
    pdf.ln(5)
    
    # --- 4. PORTFOLIO SELECTION ---
    pdf.set_font("Helvetica", 'B', 14); pdf.cell(200, 10, txt="4. Recommended Portfolio", ln=True)
    pdf.set_font("Helvetica", size=12)
    for index, row in rec_stocks.iterrows():
        stock = row.get('Stock_Name', 'Unknown')
        sec = row.get('Sector', 'Unknown')
        pdf.cell(200, 10, txt=f"- {stock} ({sec})", ln=True)
    pdf.ln(5)

    # --- 5. RISK MANAGEMENT ---
    pdf.set_font("Helvetica", 'B', 14); pdf.cell(200, 10, txt="5. Risk Management", ln=True)
    pdf.set_font("Helvetica", size=12)
    pdf.cell(200, 10, txt=f"Emergency Fund Needed: {safe_text(format_indian(emergency_fund))}", ln=True)
    pdf.cell(200, 10, txt=f"Term Insurance Needed: {safe_text(format_indian(insurance_needed))}", ln=True)
    pdf.ln(5)

    # --- 6. AI STRATEGIC INSIGHTS (New) ---
    pdf.set_font("Helvetica", 'B', 14); pdf.cell(200, 10, txt="6. AI Strategic Insights", ln=True)
    pdf.set_font("Helvetica", size=12)
    for tip in strategic_tips:
        # Clean emojis because FPDF sometimes struggles with them, or use a font that supports them.
        # Simple cleanup for standard FPDF:
        clean_tip = tip.encode('latin-1', 'ignore').decode('latin-1') 
        pdf.multi_cell(0, 8, txt=f"- {clean_tip}")
    
    pdf.ln(10)
    pdf.set_font("Helvetica", 'I', 10)
    pdf.multi_cell(0, 10, "Disclaimer: This is an academic project prototype. Investments are subject to market risk. Please consult a SEBI registered advisor before investing.")

    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 5. USER INTERFACE (SIDEBAR)
# ==========================================
st.sidebar.title("📝 User Details")
st.sidebar.caption(f"Market Data Updated: {datetime.date.today().strftime('%d %b %Y')}")

# --- 1. DEMOGRAPHICS FOR RESEARCH ---
st.sidebar.subheader("👤 Personal Info (For Research)")
user_name = st.sidebar.text_input("First Name", placeholder="e.g., Amit")
user_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
user_city = st.sidebar.selectbox("Location Type", ["Metro City", "Tier 2 City", "Tier 3 / Rural"])
dependents = st.sidebar.number_input("Number of Dependents (Kids/Parents)", 0, 10, 0)

st.sidebar.markdown("---")

# --- 2. FINANCIAL BASELINE ---
st.sidebar.subheader("💰 Income & Expenses")
age = st.sidebar.number_input("Current Age", 18, 80, 25)
retire_age = st.sidebar.number_input("Retirement Age", 40, 90, 60)
income = st.sidebar.number_input("Monthly Income (₹)", 10000, 1000000, 50000, step=5000)
expenses = st.sidebar.number_input("Monthly Expenses (₹)", 5000, 500000, 20000, step=1000)

# Smart Tax Logic
annual_income = income * 12
tax_threshold = 775000 
tax_default = True if annual_income > tax_threshold else False

st.sidebar.markdown("---")

# --- 3. LOAN MANAGEMENT ---
st.sidebar.subheader("🏦 Loan Management")
st.sidebar.caption("Add your active loans here:")

default_loans = pd.DataFrame([
    {"Loan Name": "Car Loan", "EMI Amount (₹)": 15000, "Interest Rate (%)": 9.5, "Years Remaining": 3},
    {"Loan Name": "Home Loan", "EMI Amount (₹)": 0, "Interest Rate (%)": 8.5, "Years Remaining": 15},
])

loan_df = st.sidebar.data_editor(
    default_loans, 
    num_rows="dynamic",
    column_config={
        "Loan Name": st.column_config.TextColumn("Loan Name"),
        "EMI Amount (₹)": st.column_config.NumberColumn("EMI (₹)", min_value=0, step=1000),
        "Interest Rate (%)": st.column_config.NumberColumn("Rate (%)", min_value=0.0, max_value=50.0, step=0.1),
        "Years Remaining": st.column_config.NumberColumn("Years Left", min_value=0.1, max_value=30.0, step=0.5)
    },
    width='stretch'
)

st.sidebar.markdown("---")

# --- 4. CURRENT ASSETS & PROTECTION ---
st.sidebar.subheader("🛡️ Current Assets & Protection")

has_health_ins = st.sidebar.checkbox("✅ Health Insurance (Self/Family)")
has_term_ins = st.sidebar.checkbox("✅ Term Life Insurance")

existing_wealth = st.sidebar.number_input("Total Existing Investments (₹)", 0, 100000000, 0, step=50000, help="Total value of FDs, Stocks, Gold, PFs, etc.")

existing_investments = st.sidebar.multiselect(
    "Where are you currently invested?",
    ["Mutual Funds", "Stocks/Equity", "Gold", "Fixed Deposits (FD)", "Real Estate", "Crypto", "PPF/EPF"],
    default=[]
)

need_tax_saving = st.sidebar.checkbox("I need Tax Saving (80C) 📝", value=tax_default)

st.sidebar.markdown("---")

# --- 5. GOALS & PSYCHOLOGY (NEW) ---
st.sidebar.subheader("🎯 Your Life Goals")
st.sidebar.caption("Add your short and long-term goals:")

default_goals = pd.DataFrame([
    {"Goal": "Car", "Cost (₹)": 800000, "Years Away": 3},
    {"Goal": "Dream Home", "Cost (₹)": 5000000, "Years Away": 10},
])

# Create the interactive table
goal_df = st.sidebar.data_editor(
    default_goals, 
    num_rows="dynamic",
    column_config={
        "Goal": st.column_config.SelectboxColumn(
            "Goal Type",
            help="Select or type a goal",
            options=["House", "Car", "Bike", "Marriage", "Business", "Vacation", "Emergency Fund", "Other"],
            required=True,
        ),
        "Cost (₹)": st.column_config.NumberColumn("Cost (₹)", min_value=10000, step=50000),
        "Years Away": st.column_config.NumberColumn("Years", min_value=1, max_value=50, step=1)
    },
    width='stretch'
)

step_up_rate = st.sidebar.slider("Annual Salary/SIP Increase (%)", 0, 20, 7)

fin_fear = st.sidebar.selectbox("What is your biggest financial fear?", [
    "Inflation eating my savings", 
    "Job loss / Income stop", 
    "Major Medical Emergency", 
    "Outliving my money in retirement",
    "Market crashes"
])

st.sidebar.markdown("---")

# --- 6. RISK PROFILE ---
st.sidebar.subheader("🧠 Risk Profile")
q1 = st.sidebar.radio("1. What is your primary goal?", ("Avoid Loss (1)", "Stable Income (3)", "Grow Wealth (5)"))
q2 = st.sidebar.radio("2. Investment Time Period?", ("Less than 3 Years (1)", "3-7 Years (3)", "More than 7 Years (5)"))
q3 = st.sidebar.radio("3. Reaction to Market Crash?", ("Panic Sell (1)", "Hold & Wait (3)", "Buy More (5)"))

def calculate_score(q1, q2, q3):
    score = 0
    if "Avoid" in q1: score += 1
    elif "Income" in q1: score += 3
    else: score += 5
    if "Less" in q2: score += 1
    elif "3-7" in q2: score += 3
    else: score += 5
    if "Sell" in q3: score += 1
    elif "Hold" in q3: score += 3
    else: score += 5
    return score * 2

current_risk_score = calculate_score(q1, q2, q3)

# Notice: Database buttons deleted entirely!

# ==========================================
# 6. MAIN EXECUTION (FULL & FIXED)
# ==========================================
st.title("🤖 AI-Powered Investment Advisor")
with st.expander("ℹ️  **System Architecture (Click to Expand)**"):
    st.write("1. **Logic:** Random Forest for Risk Profiling.")
    st.write("2. **Selection:** Sharpe Ratio & K-Means Clustering.")
    st.write("3. **Projection:** Dynamic Cash Flow Engine (Takes Loan End Dates into account).")

st.markdown("---")

# --- NEW: AT-A-GLANCE DASHBOARD ---
# Calculate EMI just for the screen display
display_emi = loan_df['EMI Amount (₹)'].sum() if not loan_df.empty else 0
display_surplus = income - expenses - display_emi

col1, col2, col3, col4 = st.columns(4)
col1.metric("Monthly Income", f"₹{int(income):,}")
col2.metric("Monthly Expenses", f"₹{int(expenses):,}")

# The logic you asked for: Hide the number if Debt is 0!
if display_emi > 0:
    col3.metric("Debt Pay (EMI)", f"₹{int(display_emi):,}")
else:
    col3.success("🎉 Debt Free!")

col4.metric("Available to Invest", f"₹{int(display_surplus):,}")
st.markdown("---")

if st.button("Generate My Investment Plan 🚀", type="primary"):
    
    # 0. BLOCKING ERROR CHECK
    if retire_age <= age:
        st.error("❌ Invalid Age: Retirement Age must be greater than Current Age.")
        st.stop()
        
# --- NEW: SILENT DATA COLLECTION FOR MBA REPORT ---
    
    # 1. Initialize empty defaults (using the exact names you want)
    goal1_name, goal1_cost, goal1_years = "None", 0, 0
    goal2_name, goal2_cost, goal2_years = "None", 0, 0
    goal3_name, goal3_cost, goal3_years = "None", 0, 0

    # 2. Safely extract up to the first 3 goals from the table
    if not goal_df.empty:
        valid_goals = goal_df.dropna().head(3) # Take only the top 3
        goals_list = valid_goals.to_dict('records')
        
        if len(goals_list) > 0:
            goal1_name = str(goals_list[0]['Goal'])
            goal1_cost = int(goals_list[0]['Cost (₹)'])
            goal1_years = int(goals_list[0]['Years Away'])
        if len(goals_list) > 1:
            goal2_name = str(goals_list[1]['Goal'])
            goal2_cost = int(goals_list[1]['Cost (₹)'])
            goal2_years = int(goals_list[1]['Years Away'])
        if len(goals_list) > 2:
            goal3_name = str(goals_list[2]['Goal'])
            goal3_cost = int(goals_list[2]['Cost (₹)'])
            goal3_years = int(goals_list[2]['Years Away'])

    # 3. Build the massive, clean row for Google Sheets
    user_data_row = [
        datetime.date.today().strftime("%Y-%m-%d"), 
        str(user_name), str(user_gender), str(user_city), int(dependents),
        int(age), int(retire_age), int(income), int(expenses), int(display_emi), 
        bool(has_health_ins), bool(has_term_ins), int(existing_wealth),
        
        # --- NEW: TOP 3 GOALS SEPARATED ---
        str(goal1_name), int(goal1_cost), int(goal1_years),
        str(goal2_name), int(goal2_cost), int(goal2_years),
        str(goal3_name), int(goal3_cost), int(goal3_years),
        
        int(step_up_rate), str(fin_fear), int(current_risk_score)
    ]
    
    # Send it to Google Sheets in the background!
    save_to_google_sheets(user_data_row)

    # --- 1. DYNAMIC CALCULATION ENGINE ---
    years_to_invest = retire_age - age
    
    # Default return (Safety init)
    exp_return = 8.0 
    
    # Run the Advanced Calculation first
    future_val, real_val, total_invested, current_total_emi = calculate_future_wealth_dynamic(
        income, expenses, loan_df, exp_return, years_to_invest, step_up_rate
    )
    
    # Define Surplus
    monthly_surplus = income - expenses - current_total_emi
    
    # --- BUG FIX: CHECK DEBT STATUS SAFELY ---
    # Only look at loans where the user actually has an EMI greater than 0
    active_loans = loan_df[loan_df['EMI Amount (₹)'] > 0] if not loan_df.empty else pd.DataFrame()
    
    if not active_loans.empty and (active_loans['Interest Rate (%)'] > 12.0).any():
         debt_check = "PRIORITY: PAY DEBT"
    else:
         debt_check = "SAFE"
    
    # --- VC FEATURE: FINANCIAL WELLNESS SCORE ---
    st.header("🏆 Financial Wellness Score")
    
    if monthly_surplus > 0: e_status = "Safe"
    else: e_status = "Unsafe"
    
    wellness_score = calculate_wellness_score(
        income, expenses, current_total_emi, e_status,
        has_health_ins, has_term_ins, existing_investments
    )
    
    col_score1, col_score2 = st.columns([1, 3])
    with col_score1:
        st.metric("Your FinFit Score", f"{wellness_score}/100")
    with col_score2:
        if wellness_score >= 80:
            st.success("🌟 Excellent! You are a Financial Master.")
            st.progress(wellness_score)
        elif wellness_score >= 50:
            st.warning("⚠️ Good start, but room for improvement.")
            st.progress(wellness_score)
        else:
            st.error("🚨 Critical: Focus on reducing debt and getting insurance.")
            st.progress(wellness_score)

    st.markdown("---")
    
    # 2. SAFETY REPORT
    st.header("1️⃣ Safety & Risk Management")
    
    # We change this to 3 columns to make room for Health Insurance!
    col_safe1, col_safe2, col_safe3 = st.columns(3) 
    
    emergency_fund = expenses * 6
    insurance_needed = (income * 12) * 20
    health_cover_needed = 1000000 # Standard baseline recommendation (₹10 Lakhs)
    
    col_safe1.metric("🛡️ Emergency Fund", format_indian(emergency_fund), "Liquid Assets")
    col_safe2.metric("🏥 Term Life Cover", format_indian(insurance_needed), "Recommended Coverage")
    
    # NEW: Dynamic Health Insurance Display
    if has_health_ins:
        col_safe3.metric("🩺 Health Cover", "Active ✅", "Family Protected")
    else:
        col_safe3.metric("🩺 Health Cover", format_indian(health_cover_needed), "⚠️ Missing (High Risk)")
    
    # 3. STRATEGY DECISION
    if debt_check == "PRIORITY: PAY DEBT":
        st.error("⚠️ HIGH INTEREST LOAN DETECTED (>12%)")
        st.warning("Recommended: Use your surplus to prepay high-interest loans first.")
    
    elif monthly_surplus <= 0:
        st.error("⚠️ Expenses + EMIs exceed Income. No surplus available.")
        st.write(f"Income: {income} | Expenses: {expenses} | Total EMI: {current_total_emi}")
        
    else:
        # 4. ML RISK PROFILING
        prediction = risk_model.predict([[age, income, current_risk_score]])
        user_profile = prediction[0]
        
        st.markdown("---")
        st.header(f"2️⃣ Strategy: {user_profile} Investor")
        
        # This is the logic that defines 'cluster_target'
        if user_profile == "Conservative":
            allocation = {"Equity": 20, "Debt Funds": 60, "Gold": 20}
            exp_return = 8.0 
            cluster_target = "Safe"
        elif user_profile == "Moderate":
            allocation = {"Equity": 50, "Debt Funds": 30, "Gold": 20}
            exp_return = 10.0
            cluster_target = "Moderate"
        else:
            allocation = {"Equity": 70, "Debt Funds": 20, "Gold": 10}
            exp_return = 12.0
            cluster_target = "Risky"
            
        # Re-Calculate Wealth with the CORRECT Expected Return
        future_val, real_val, total_invested, _ = calculate_future_wealth_dynamic(
            income, expenses, loan_df, exp_return, years_to_invest, step_up_rate
        )

        c1, c2 = st.columns([1, 2])
        with c1:
            fig1, ax1 = plt.subplots(figsize=(3,3))
            ax1.pie(allocation.values(), labels=allocation.keys(), autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
            st.pyplot(fig1)
        with c2:
            st.info(f"**Analysis:** Based on Age ({age}) and Risk Score ({current_risk_score}).")
            st.write(f"📈 Expected Annual Return: **{exp_return}%**")
        
        # 5. MULTI-GOAL ANALYSIS
        st.markdown("---")
        st.subheader("🎯 Life Goals Analysis")
        
        if goal_df.empty or goal_df['Cost (₹)'].sum() == 0:
            st.info("No specific goals added. Focus on building general wealth!")
        else:
            # Loop through every goal the user put in the table
            for index, row in goal_df.dropna().iterrows():
                g_name = row['Goal']
                g_cost = row['Cost (₹)']
                g_years = row['Years Away']
                
                # A. INFLATION ADJUSTMENT
                inflated_cost = g_cost * ((1 + 0.06) ** g_years)
                
                # B. SMART PROJECTION (Wealth at that specific year)
                goal_projected_val, _, _, _ = calculate_future_wealth_dynamic(
                    income, expenses, loan_df, exp_return, g_years, step_up_rate
                )
                
                # C. DISPLAY IN AN EXPANDER (Clean UI)
                with st.expander(f"📌 {g_name} in {int(g_years)} Years (Need: {format_indian(inflated_cost)})"):
                    c_goal1, c_goal2, c_goal3 = st.columns(3)
                    c_goal1.metric("Current Cost", format_indian(g_cost))
                    c_goal2.metric("Inflated Cost", format_indian(inflated_cost), "6% Inflation")
                    c_goal3.metric("Projected Wealth", format_indian(goal_projected_val))
                    
                    if goal_projected_val >= inflated_cost:
                        st.success(f"✅ On Track! Your projected wealth safely covers this {g_name}.")
                    else:
                        st.warning(f"❌ Shortfall Detected. You will be short by {format_indian(inflated_cost - goal_projected_val)}.")
                        st.info("💡 Tip: Set up a dedicated Mutual Fund SIP specifically for this goal.")

        # --- 6. WEALTH PROJECTION (TABBED INTERFACE) ---
        st.markdown("---")
        st.header("3️⃣ Wealth Projection")
        
        tab1, tab2 = st.tabs(["📊 Standard Projection", "🎛️ What-If Scenarios"])
        
        with tab1:
            # This is your existing chart code
            st.caption(f"Based on {exp_return}% Return and {step_up_rate}% Annual Step-up.")
            chart_data = get_projection_data_dynamic(income, expenses, loan_df, exp_return, years_to_invest, step_up_rate)
            st.area_chart(chart_data, color=["#A9A9A9", "#4CAF50"])
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Principal", format_indian(total_invested))
            m2.metric("Wealth Gained", format_indian(future_val - total_invested))
            m3.metric("Total Corpus", format_indian(future_val))

        with tab2:
            st.write("### 🔮 Simulate Future Scenarios")
            col_sim1, col_sim2 = st.columns(2)
            
            # Temporary sliders just for this tab
            sim_return = col_sim1.slider("Hypothetical Return (%)", 5.0, 25.0, exp_return, 0.5)
            sim_stepup = col_sim2.slider("Hypothetical Step-Up (%)", 0, 20, step_up_rate, 1)
            
            # Recalculate specifically for simulation
            sim_future, _, _, _ = calculate_future_wealth_dynamic(
                income, expenses, loan_df, sim_return, years_to_invest, sim_stepup
            )
            
            diff = sim_future - future_val
            
            st.metric("Simulated Corpus", format_indian(sim_future), delta=format_indian(diff))
            st.caption("See how small changes in Return or Step-up drastically change your final wealth.")
            
            sim_chart = get_projection_data_dynamic(income, expenses, loan_df, sim_return, years_to_invest, sim_stepup)
            st.line_chart(sim_chart['Wealth Created (Interest)'])
        
        # chart_data = get_projection_data_dynamic(income, expenses, loan_df, exp_return, years_to_invest, step_up_rate)
        # st.area_chart(chart_data, color=["#A9A9A9", "#4CAF50"]) 
        

        # --- NEW: RETIREMENT LIFESTYLE CHECK (SWP Logic) ---
        st.markdown("#### 🛌3.1 Can I retire comfortably?")
        
        # 1. Calculate Future Monthly Expenses (Inflation Adjusted)
        # We assume expenses grow at 6% inflation until retirement
        future_monthly_expenses = expenses * ((1 + 0.06) ** years_to_invest)
        
        # 2. Calculate Safe Monthly Passive Income from Corpus (SWP)
        # Standard Rule: You can safely withdraw 6% of your corpus annually without depleting it too fast
        safe_withdrawal_rate = 0.06
        monthly_passive_income = (future_val * safe_withdrawal_rate) / 12
        
        # 3. Compare
        col_life1, col_life2, col_life3 = st.columns(3)
        col_life1.metric("Future Monthly Expense", format_indian(int(future_monthly_expenses)), help="Your current expenses adjusted for inflation.")
        col_life2.metric("Your Passive Income", format_indian(int(monthly_passive_income)), help="Monthly income generated from your corpus @ 6% withdrawal.")
        
        surplus_deficit = monthly_passive_income - future_monthly_expenses
        
        if surplus_deficit >= 0:
            col_life3.success(f"✅ Freedom! Surplus: {format_indian(int(surplus_deficit))}/mo")
            st.caption("🎉 Result: You can maintain your current lifestyle comfortably.")
        else:
            col_life3.error(f"❌ Shortfall: {format_indian(int(abs(surplus_deficit)))}/mo")
            st.caption("⚠️ Warning: Your corpus isn't enough to cover inflation-adjusted expenses. Increase SIP!")
            
        st.markdown("---")
        
        
        # --- 7. PORTFOLIO TABLE (UPGRADED WITH LIVE DATA) ---
        st.markdown("---")
        st.header("4️⃣ Recommended Portfolio & Sector Analysis")
        
        rec_stocks = get_diversified_stocks(cluster_target, need_tax_saving)
        
        with st.spinner("📈 Fetching Live Market Data for your portfolio..."):
            # Fetch Nifty 50 for the header
            nifty_data = get_live_prices(["^NSEI"]) 
            
            # Fetch prices for the exact stocks recommended
            symbols_to_fetch = rec_stocks['Symbol'].dropna().tolist()
            live_stock_prices = get_live_prices(symbols_to_fetch)
            
        col_live, col_dummy = st.columns([2, 1])
        if nifty_data:
            nifty_val = list(nifty_data.values())[0]
            col_live.metric("🔴 Live Market Status (Nifty 50)", f"₹{nifty_val:,.2f}", delta="Real-time update")
        else:
            col_live.caption("Live data unavailable (Offline Mode)")
        
        # Fund Manager Feature: Beta
        avg_beta = rec_stocks['Volatility_Beta'].mean()
        if avg_beta < 0.8: beta_msg = "Low Volatility (Defensive)"
        elif avg_beta < 1.2: beta_msg = "Market Standard (Balanced)"
        else: beta_msg = "High Volatility (Aggressive)"
        st.caption(f"🛡️ **Portfolio Beta:** {avg_beta:.2f} ({beta_msg})")

        # Sector Chart
        col_stock_table, col_sector_chart = st.columns([3, 2])
        
        with col_stock_table:
            st.caption("Selected High-Quality Stocks:")
            display_df = rec_stocks.copy()
            
            # Map the fetched live prices to the dataframe
            if live_stock_prices:
                # We strip the '.NS' suffix when matching back to the dataframe
                display_df['Live Price'] = display_df['Symbol'].apply(
                    lambda x: f"₹{live_stock_prices.get(str(x) + '.NS', live_stock_prices.get(x, 0)):.2f}"
                )
            else:
                display_df['Live Price'] = "Offline"
                
            display_df = display_df[['Stock_Name', 'Sector', 'Annual_Return', 'Volatility_Beta', 'Live Price']]
            display_df.columns = ['Stock Name', 'Category', '3Y Return (%)', 'Risk Score', 'Live Price']
            
            st.dataframe(display_df, hide_index=True, width='stretch')

        with col_sector_chart:
            st.caption("Sector Exposure:")
            sector_counts = rec_stocks['Sector'].value_counts()
            fig_sec, ax_sec = plt.subplots(figsize=(3,3))
            ax_sec.pie(sector_counts, labels=sector_counts.index, autopct='%1.0f%%', startangle=90, colors=['#4CAF50', '#FFC107', '#2196F3', '#9C27B0', '#FF5722'])
            centre_circle = plt.Circle((0,0),0.70,fc='white')
            fig_sec.gca().add_artist(centre_circle)
            st.pyplot(fig_sec)
        
        equity_share = int((monthly_surplus * allocation['Equity']) / 100)
        
        st.success(f"💡 **Action:** Setup SIP of **{format_indian(equity_share)}** divided among these funds.")
        st.info("🗓️ **Review:** Rebalance portfolio every 12 months.")
        
        # ... (After the Portfolio & Sector Chart Section) ...

        # --- 8. AI STRATEGIC INSIGHTS (NEW) ---
        st.markdown("---")
        st.header("5️⃣ AI Strategic Insights")
        
        # Add a visual loading spinner so we know it's working
        with st.spinner("🤖 AI is analyzing your financial profile... (This may take 30-60 seconds on local hardware)"):
            strategic_tips = generate_smart_strategy(user_profile, age, wellness_score, debt_check, monthly_surplus)
        
        # Display as a clean list
        for tip in strategic_tips:
            st.info(tip)
        
        # Generate the tips
        strategic_tips = generate_smart_strategy(user_profile, age, wellness_score, debt_check, monthly_surplus)
        
            
        # Add a static "Pro Tip" for everyone
        with st.expander("💡 Pro Tip: The Rule of 72"):
            st.write(f"""
            The **Rule of 72** is a quick way to calculate how long it will take to double your money.
            
            * **Formula:** 72 ÷ Interest Rate = Years to Double
            * **For You:** At {exp_return}% expected return, your money will double approx every **{72/exp_return:.1f} years**.
            """)


        # --- 9. DOWNLOAD REPORT (Renumbered to 6) ---
        st.markdown("---")
        st.header("6️⃣ Download Report")
        
        pdf_data = create_pdf_report(
            user_profile, 
            monthly_surplus, 
            future_val, 
            allocation, 
            rec_stocks, 
            emergency_fund, 
            insurance_needed, 
            current_total_emi,
            wellness_score,
            strategic_tips,         
            monthly_passive_income, 
            future_monthly_expenses
        )
        st.download_button("📄 Download PDF Report", data=pdf_data, file_name="Financial_Plan.pdf", mime="application/pdf")

else:
    st.info("👈 Enter details in the sidebar to generate your plan.")

# Footer
st.markdown("---")
st.caption("⚠️ **Disclaimer:** Academic Project. Not Investment Advice.")

