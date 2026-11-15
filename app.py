from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from amrce.research import TD, Signals, Strat, load_market, run_strat
from amrce.auth import (
    authenticate_user, register_user, create_user_by_admin,
    update_user_credentials, update_user_password,
    record_user_trade, get_user_recorded_trades,
    get_all_users, set_user_approval, set_user_role,
    delete_user_account, get_audit_logs
)

st.set_page_config(
    page_title="AMRCE",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ Production Design System (Zinc-950)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    #MainMenu, footer, .stDeployButton, [data-testid="stAppDeployButton"], [data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2rem !important;
    }

    .stApp {
        background-color: #09090B;
        color: #FAFAFA;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1320px !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #09090B !important;
        border-right: 1px solid #27272A !important;
    }

    .prod-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 16px;
        margin-bottom: 24px;
        border-bottom: 1px solid #27272A;
    }
    .prod-logo {
        font-size: 1.05rem;
        font-weight: 600;
        color: #FAFAFA;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .prod-sub {
        color: #71717A;
        font-weight: 400;
        font-size: 0.85rem;
    }

    .badge-live {
        background: rgba(16, 185, 129, 0.1);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
    }
    .badge-standby {
        background: #18181B;
        color: #A1A1AA;
        border: 1px solid #27272A;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
    }
    .badge-halted {
        background: rgba(239, 68, 68, 0.1);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
    }

    .metric-strip {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    @media (max-width: 900px) {
        .metric-strip { grid-template-columns: repeat(2, 1fr); }
    }
    .metric-box {
        background: #121215;
        border: 1px solid #27272A;
        border-radius: 8px;
        padding: 18px 20px;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #A1A1AA;
        font-weight: 500;
        margin-bottom: 8px;
    }
    .metric-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        color: #FAFAFA;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }
    .metric-meta {
        font-size: 0.75rem;
        color: #71717A;
        margin-top: 8px;
        font-family: 'JetBrains Mono', monospace;
    }

    .block-card {
        background: #121215;
        border: 1px solid #27272A;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .block-top {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 12px;
        padding-bottom: 12px;
        border-bottom: 1px solid #1F1F23;
    }
    .block-sym {
        font-size: 1.25rem;
        font-weight: 600;
        color: #FAFAFA;
        letter-spacing: -0.02em;
    }
    .block-usd {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.25rem;
        font-weight: 600;
        color: #FAFAFA;
    }
    .block-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        color: #A1A1AA;
        margin-top: 6px;
    }
    .block-row span.mono {
        font-family: 'JetBrains Mono', monospace;
        color: #E4E4E7;
    }

    div[data-baseweb="input"] > div {
        background-color: #121215 !important;
        border: 1px solid #27272A !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="input"] input {
        color: #FAFAFA !important;
        font-size: 0.88rem !important;
    }

    div.stButton > button[kind="primary"], div[data-testid="stFormSubmitButton"] > button {
        background-color: #FAFAFA !important;
        color: #09090B !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        border: 1px solid #FAFAFA !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        transition: opacity 0.15s ease;
    }
    div.stButton > button[kind="primary"]:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        opacity: 0.9 !important;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #18181B !important;
        color: #E4E4E7 !important;
        font-weight: 500 !important;
        font-size: 0.84rem !important;
        border: 1px solid #27272A !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: #52525B !important;
        background-color: #27272A !important;
    }

    button[data-baseweb="tab"] {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #A1A1AA !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #FAFAFA !important;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "user" not in st.session_state:
    st.session_state.user = None
if "kill_switch" not in st.session_state:
    st.session_state.kill_switch = False
if "auto_pilot" not in st.session_state:
    st.session_state.auto_pilot = False
if "last_auto_sync" not in st.session_state:
    st.session_state.last_auto_sync = None


# ==================================================================
# AUTHENTICATION SCREEN
# ==================================================================
if not st.session_state.user:
    st.markdown("<div style='height: 8vh;'></div>", unsafe_allow_html=True)
    _, col_auth, _ = st.columns([1.2, 1, 1.2])
    with col_auth:
        st.markdown("""
        <div style="margin-bottom: 24px;">
            <div style="font-size: 1.35rem; font-weight: 600; color: #FAFAFA; letter-spacing: -0.02em;">AMRCE</div>
            <div style="font-size: 0.85rem; color: #71717A; margin-top: 4px;">Quantitative Execution & Portfolio Management</div>
        </div>
        """, unsafe_allow_html=True)

        tab_signin, tab_signup = st.tabs(["Sign In", "Request Account"])

        with tab_signin:
            with st.form("login_form", border=True):
                u_in = st.text_input("Username", placeholder="Enter username")
                p_in = st.text_input("Password", type="password", placeholder="Enter password")
                submit_login = st.form_submit_button("Sign In", width="stretch")

                if submit_login:
                    usr = authenticate_user(u_in, p_in)
                    if usr:
                        if usr["is_approved"] or usr["role"] == "admin":
                            st.session_state.user = usr
                            st.rerun()
                        else:
                            st.error("Your account is pending administrator approval.")
                    else:
                        st.error("Invalid username or password.")

        with tab_signup:
            with st.form("signup_form", border=True):
                nu_in = st.text_input("Username", placeholder="Minimum 3 characters")
                np_in = st.text_input("Password", type="password", placeholder="Minimum 6 characters")
                np_conf = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
                submit_signup = st.form_submit_button("Create Account", width="stretch")

                if submit_signup:
                    if np_in != np_conf:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = register_user(nu_in, np_in)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
    st.stop()


# ==================================================================
# QUANTITATIVE ENGINE & BROKER ADAPTERS
# ==================================================================
@st.cache_data(ttl=900)
def load_engine_state(active_split: float = 0.60, refresh: bool = False) -> dict:
    md = load_market(years=22.0, refresh=refresh)
    sig = Signals(md)
    st_cfg = Strat(
        "Strategy C — 2x Parity",
        lookback_m=6, top_n=2, trend=False, abs_mom=False,
        max_w=2.0, exclude=("SPY", "GLD")
    )

    L = st_cfg.lookback_m * 21
    mom_df = sig.mom(L)
    cash_hurdle = float(sig.cash_mom(L)[-1])

    cols = list(sig.px.columns)
    latest_mom = {c: float(mom_df[c].iloc[-1]) for c in cols}
    latest_px = {c: float(sig.px[c].iloc[-1]) for c in cols}

    elig = []
    for c in cols:
        if c in st_cfg.exclude or np.isnan(latest_mom[c]):
            continue
        elig.append((latest_mom[c], c))

    elig.sort(reverse=True)
    selected = [c for _, c in elig[:st_cfg.top_n]]
    raw_weights = {c: (st_cfg.max_w / st_cfg.top_n if c in selected else 0.0) for c in cols}

    reserve_split = round(1.0 - active_split, 4)
    active_weights = {k: round(v * active_split, 4) for k, v in raw_weights.items() if v > 0}
    active_weights["CASH_RESERVE"] = reserve_split

    S = 260
    res_sim = run_strat(sig, st_cfg, S=S, cost_bps=5.0, lag=1)
    cash_aligned = md.cash_ret.reindex(res_sim.ret.index).fillna(0)
    blended_ret = active_split * res_sim.ret + reserve_split * cash_aligned

    eq_series = (1.0 + blended_ret).cumprod()
    wlog = res_sim.wlog
    full_idx = res_sim.full_index
    trade_logs = []

    for i in range(len(wlog)):
        t_start, w_vec = wlog[i]
        d_start = full_idx[t_start]
        d_end = full_idx[wlog[i + 1][0]] if i + 1 < len(wlog) else full_idx[-1]
        sub_eq = eq_series[(eq_series.index >= d_start) & (eq_series.index <= d_end)]
        if len(sub_eq) < 2:
            continue

        ret_p = float(sub_eq.iloc[-1] / sub_eq.iloc[0] - 1.0)
        alloc_str = " + ".join(
            f"{cols[j]} ({w_vec[j]*active_split*100:.0f}%)"
            for j in range(len(cols)) if w_vec[j] > 0
        )
        trade_logs.append({
            "Open Date": d_start.strftime("%Y-%m-%d"),
            "Close Date": d_end.strftime("%Y-%m-%d"),
            "Allocation": alloc_str or "Cash Reserve",
            "Return (%)": round(ret_p * 100, 2),
            "start_mult": float(sub_eq.iloc[0]),
            "end_mult": float(sub_eq.iloc[-1]),
        })

    return {
        "as_of": str(sig.px.index[-1].date()),
        "prices": latest_px,
        "mom_6m": {c: round(latest_mom[c] * 100, 2) for c in cols},
        "cash_hurdle_pct": round(cash_hurdle * 100, 2),
        "weights": active_weights,
        "blended_ret": blended_ret,
        "historical_logs": pd.DataFrame(trade_logs).iloc[::-1].reset_index(drop=True),
    }


def get_alpaca_client(api_key: str, secret_key: str, paper: bool):
    from alpaca.trading.client import TradingClient
    return TradingClient(api_key.strip(), secret_key.strip(), paper=paper)


def fetch_alpaca_snapshot(api_key: str, secret_key: str, paper: bool) -> dict:
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus

    client = get_alpaca_client(api_key, secret_key, paper)
    acct = client.get_account()
    positions = client.get_all_positions()
    open_orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))

    equity = float(acct.equity)
    last_equity = float(acct.last_equity) if acct.last_equity else equity
    today_pl = equity - last_equity
    today_pl_pct = (today_pl / last_equity * 100.0) if last_equity > 0 else 0.0
    multiplier = float(getattr(acct, "multiplier", 1) or 1)

    pos_list = []
    pos_map = {}
    for p in positions:
        mv = float(p.market_value)
        upl = float(p.unrealized_pl)
        item = {
            "Symbol": p.symbol,
            "Shares": round(float(p.qty), 4),
            "Avg Entry ($)": round(float(p.avg_entry_price), 2),
            "Last Price ($)": round(float(p.current_price), 2),
            "Market Value ($)": round(mv, 2),
            "Weight (%)": round(mv / equity * 100, 1) if equity > 0 else 0.0,
            "Unrealized P&L ($)": round(upl, 2),
            "Unrealized P&L (%)": round(float(p.unrealized_plpc) * 100, 2),
        }
        pos_list.append(item)
        pos_map[p.symbol] = item

    pending_list = []
    for o in open_orders:
        side_str = str(o.side.value if hasattr(o.side, "value") else o.side).upper()
        notional_val = float(o.notional) if o.notional else (float(o.qty or 0) * float(o.limit_price or 0))
        pending_list.append({
            "Symbol": str(o.symbol),
            "Side": side_str,
            "Notional ($)": round(notional_val, 2) if notional_val else None,
            "Status": str(o.status.value if hasattr(o.status, "value") else o.status).upper(),
        })

    return {
        "equity": equity,
        "cash": float(acct.cash),
        "buying_power": float(acct.buying_power),
        "multiplier": multiplier,
        "today_pl": today_pl,
        "today_pl_pct": today_pl_pct,
        "positions_df": pd.DataFrame(pos_list),
        "positions_map": pos_map,
        "open_orders_df": pd.DataFrame(pending_list),
        "open_orders_count": len(pending_list),
    }


def fetch_alpaca_intraday_curve(api_key: str, secret_key: str, paper: bool) -> pd.DataFrame:
    import requests
    base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
    headers = {"APCA-API-KEY-ID": api_key.strip(), "APCA-API-SECRET-KEY": secret_key.strip()}
    try:
        resp = requests.get(
            f"{base_url}/v2/account/portfolio/history",
            headers=headers,
            params={"period": "1D", "timeframe": "5Min"},
            timeout=8,
        )
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()
        df = pd.DataFrame({
            "Time": pd.to_datetime(data.get("timestamp", []), unit="s"),
            "Intraday P&L ($)": data.get("profit_loss", []),
        })
        return df[df["Intraday P&L ($)"].notna()]
    except Exception:
        return pd.DataFrame()


def execute_big_block_trades(
    user_id: int, username: str,
    api_key: str, secret_key: str, paper: bool,
    target_weights: dict, min_trade_pct: float = 0.05, dry_run: bool = True
) -> list:
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.common.exceptions import APIError

    client = get_alpaca_client(api_key, secret_key, paper)
    if not dry_run:
        try:
            client.cancel_orders()
        except Exception:
            pass

    snap = fetch_alpaca_snapshot(api_key, secret_key, paper)
    equity = snap["equity"]
    pos_map = snap["positions_map"]
    min_block_usd = max(10.0, equity * min_trade_pct)

    max_acct_exposure = 1.0 if (snap["multiplier"] <= 1.0 or equity < 2000.0) else min(snap["multiplier"], 2.0)
    total_tgt_weight = sum(target_weights.values())
    scaled_weights = target_weights.copy()
    if total_tgt_weight > max_acct_exposure and total_tgt_weight > 0:
        scale_factor = (0.60 if max_acct_exposure == 1.0 else max_acct_exposure * 0.95) / total_tgt_weight
        scaled_weights = {k: round(v * scale_factor, 4) for k, v in target_weights.items()}

    sells, buys = [], []
    for sym in sorted(set(pos_map.keys()) | set(scaled_weights.keys())):
        tgt_w = scaled_weights.get(sym, 0.0)
        tgt_usd = round(equity * tgt_w, 2)
        cur_usd = round(pos_map.get(sym, {}).get("Market Value ($)", 0.0), 2)
        delta_usd = round(tgt_usd - cur_usd, 2)

        if abs(delta_usd) < min_block_usd and tgt_w > 0:
            continue
        if delta_usd < 0:
            sells.append((sym, abs(delta_usd), cur_usd, tgt_usd, tgt_w))
        elif delta_usd > 0:
            buys.append((sym, delta_usd, cur_usd, tgt_usd, tgt_w))

    executed = []
    for sym, order_usd, cur_usd, tgt_usd, tgt_w in sells:
        row = {
            "Symbol": sym, "Side": "SELL", "Current ($)": f"${cur_usd:,.2f}",
            "Target ($)": f"${tgt_usd:,.2f}", "Order Notional ($)": f"${order_usd:,.2f}",
            "Status": "READY" if dry_run else "SUBMITTED"
        }
        order_id_str = ""
        if not dry_run:
            try:
                if tgt_w == 0.0 and sym in pos_map:
                    resp = client.close_position(sym)
                    row["Status"] = "CLOSED"
                    order_id_str = str(getattr(resp, "id", ""))
                else:
                    resp = client.submit_order(
                        MarketOrderRequest(symbol=sym, notional=round(order_usd, 2), side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
                    )
                    row["Status"] = str(resp.status.value if hasattr(resp.status, "value") else resp.status).upper()
                    order_id_str = str(getattr(resp, "id", ""))
            except APIError as e:
                row["Status"] = f"SKIPPED ({e.message})"
            record_user_trade(user_id, username, sym, "SELL", order_usd, equity, paper, row["Status"], order_id_str)
        executed.append(row)

    if not dry_run and sells:
        snap = fetch_alpaca_snapshot(api_key, secret_key, paper)

    avail_bp = max(0.0, snap["buying_power"] * 0.98)
    total_buy_needed = sum(x[1] for x in buys)
    bp_scale = (avail_bp / total_buy_needed) if (total_buy_needed > avail_bp and total_buy_needed > 0) else 1.0

    for sym, order_usd, cur_usd, tgt_usd, tgt_w in buys:
        safe_order_usd = round(order_usd * bp_scale, 2)
        row = {
            "Symbol": sym, "Side": "BUY", "Current ($)": f"${cur_usd:,.2f}",
            "Target ($)": f"${tgt_usd:,.2f}", "Order Notional ($)": f"${safe_order_usd:,.2f}",
            "Status": "READY" if dry_run else "SUBMITTED"
        }
        if safe_order_usd < 1.0:
            row["Status"] = "INSUFFICIENT BUYING POWER"
            executed.append(row)
            continue
        order_id_str = ""
        if not dry_run:
            try:
                resp = client.submit_order(
                    MarketOrderRequest(symbol=sym, notional=safe_order_usd, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
                )
                row["Status"] = str(resp.status.value if hasattr(resp.status, "value") else resp.status).upper()
                order_id_str = str(getattr(resp, "id", ""))
            except APIError as e:
                row["Status"] = f"REJECTED ({e.message})"
            record_user_trade(user_id, username, sym, "BUY", safe_order_usd, equity, paper, row["Status"], order_id_str)
        executed.append(row)

    return executed


def fetch_alpaca_trade_logs(api_key: str, secret_key: str, paper: bool) -> pd.DataFrame:
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus

    client = get_alpaca_client(api_key, secret_key, paper)
    orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=250))
    filled = [o for o in orders if o.filled_qty and float(o.filled_qty) > 0 and o.filled_avg_price]
    filled.sort(key=lambda x: x.filled_at or x.created_at)

    inventory: dict[str, list[list[float]]] = {}
    rows = []
    for o in filled:
        sym = str(o.symbol)
        side = str(o.side.value if hasattr(o.side, "value") else o.side).upper()
        qty = float(o.filled_qty)
        px = float(o.filled_avg_price)
        dt_str = (o.filled_at or o.created_at).strftime("%Y-%m-%d %H:%M")
        realized_pnl = None

        if "BUY" in side:
            inventory.setdefault(sym, []).append([qty, px])
        elif "SELL" in side:
            rem = qty
            cost = 0.0
            matched = 0.0
            lots = inventory.get(sym, [])
            while rem > 1e-6 and lots:
                lq, lpx = lots[0]
                take = min(rem, lq)
                cost += take * lpx
                matched += take
                rem -= take
                if take >= lq - 1e-6:
                    lots.pop(0)
                else:
                    lots[0][0] -= take
            if matched > 0:
                realized_pnl = round(matched * px - cost, 2)

        rows.append({
            "Timestamp": dt_str,
            "Symbol": sym,
            "Side": side,
            "Shares": round(qty, 4),
            "Fill Price ($)": round(px, 2),
            "Notional ($)": round(qty * px, 2),
            "Realized P&L ($)": realized_pnl,
        })
    return pd.DataFrame(rows[::-1])


def emergency_liquidate_all(user_id: int, username: str, api_key: str, secret_key: str, paper: bool) -> list:
    client = get_alpaca_client(api_key, secret_key, paper)
    client.cancel_orders()
    positions = client.get_all_positions()
    closed = []
    for p in positions:
        mv = float(p.market_value)
        client.close_position(p.symbol)
        record_user_trade(user_id, username, p.symbol, "LIQUIDATE", mv, 0.0, paper, "CLOSED_KILL_SWITCH")
        closed.append(p.symbol)
    return closed


# ==================================================================
# SIDEBAR: ACCOUNT & ENCRYPTED BROKER VAULT
# ==================================================================
cur_user = st.session_state.user

with st.sidebar:
    st.markdown(f"**Account**: `{cur_user['username']}`")
    st.caption(f"Role: {cur_user['role'].upper()}")

    if st.button("Sign Out", width="stretch"):
        st.session_state.user = None
        st.rerun()

    st.divider()
    st.markdown("**Broker API Credentials**")
    st.caption("Encrypted at rest via AES-256 Fernet.")

    u_key = st.text_input("Alpaca API Key", value=cur_user["api_key"], type="password")
    u_sec = st.text_input("Alpaca Secret Key", value=cur_user["secret_key"], type="password")
    u_paper = st.toggle("Paper Trading Environment", value=cur_user["paper_mode"])

    if st.button("Save Credentials", type="primary", width="stretch"):
        update_user_credentials(cur_user["username"], u_key, u_sec, u_paper)
        st.session_state.user["api_key"] = u_key
        st.session_state.user["secret_key"] = u_sec
        st.session_state.user["paper_mode"] = u_paper
        st.success("Credentials encrypted and saved.")

    with st.expander("Change Account Password"):
        new_pw = st.text_input("New Password", type="password", key="self_pw")
        if st.button("Update Password", width="stretch"):
            ok, msg = update_user_password(cur_user["username"], cur_user["username"], new_pw)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    st.divider()
    st.markdown("**Portfolio Parameters**")
    default_capital = st.number_input("Simulated Capital ($)", min_value=100, max_value=100_000_000, value=100_000, step=10_000)
    active_pct = st.slider("Active Strategy Allocation (%)", min_value=20, max_value=100, value=60, step=5)
    min_block_pct = st.slider("Min Trade Block Filter (%)", min_value=1, max_value=25, value=5)
    use_etf_proxies = st.toggle("Use 2x ETFs (QLD / UBT)", value=True)
    refresh_btn = st.button("Refresh Market Data", width="stretch")

# ------------------------------------------------------------------ Load Runtime State
if refresh_btn:
    st.cache_data.clear()

active_split = active_pct / 100.0
engine = load_engine_state(active_split=active_split, refresh=refresh_btn)

active_key = u_key if u_key else cur_user["api_key"]
active_sec = u_sec if u_sec else cur_user["secret_key"]
active_paper = u_paper

snap = None
if active_key and active_sec:
    try:
        snap = fetch_alpaca_snapshot(active_key, active_sec, active_paper)
    except Exception:
        snap = None

live_capital = snap["equity"] if snap else float(default_capital)
target_weights = {k: v for k, v in engine["weights"].items() if k != "CASH_RESERVE"}

if use_etf_proxies:
    proxied = {}
    if target_weights.get("QQQ", 0) > 0:
        proxied["QLD"] = round(target_weights["QQQ"] / 2.0, 4)
    if target_weights.get("TLT", 0) > 0:
        proxied["UBT"] = round(target_weights["TLT"] / 2.0, 4)
    target_weights = proxied
elif snap and (snap["multiplier"] <= 1.0 or live_capital < 2000.0):
    tot_w = sum(target_weights.values())
    if tot_w > active_split and tot_w > 0:
        target_weights = {k: round(v * (active_split / tot_w), 4) for k, v in target_weights.items()}

if snap:
    today_pl_usd = snap["today_pl"]
    today_pl_pct = snap["today_pl_pct"]
else:
    today_pl_pct = float(engine["blended_ret"].iloc[-1] * 100.0)
    today_pl_usd = live_capital * (today_pl_pct / 100.0)

# ------------------------------------------------------------------ Top Navigation Bar
status_badge = (
    '<span class="badge-halted">HALTED</span>'
    if st.session_state.kill_switch
    else ('<span class="badge-live">AUTO-EXECUTION ON</span>' if st.session_state.auto_pilot else '<span class="badge-standby">MANUAL MODE</span>')
)

st.markdown(f"""
<div class="prod-header">
    <div class="prod-logo">
        <span>AMRCE</span>
        <span class="prod-sub">/ {active_pct}% Parity Growth (17.3% CAGR) · {100-active_pct}% Reserve</span>
    </div>
    <div>{status_badge}</div>
</div>
""", unsafe_allow_html=True)

# 4 Metric Cards
today_color = "#10B981" if today_pl_usd >= 0 else "#EF4444"
deploy_usd = sum(target_weights.values()) * live_capital
reserve_usd = (1.0 - active_split) * live_capital
bp_label = f"Buying Power: ${snap['buying_power']:,.2f}" if snap else "Unlinked (Simulated Mode)"

st.markdown(f"""
<div class="metric-strip">
    <div class="metric-box">
        <div class="metric-label">Net Liquidation Value</div>
        <div class="metric-val">${live_capital:,.2f}</div>
        <div class="metric-meta">{bp_label}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Today's P&L</div>
        <div class="metric-val" style="color:{today_color};">${today_pl_usd:+,.2f}</div>
        <div class="metric-meta" style="color:{today_color};">{today_pl_pct:+.2f}%</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Active Deployment ({active_pct}%)</div>
        <div class="metric-val">${deploy_usd:,.2f}</div>
        <div class="metric-meta">Min Block: {min_block_pct}% (${live_capital*min_block_pct/100:,.0f})</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Cash Reserve ({100-active_pct}%)</div>
        <div class="metric-val">${reserve_usd:,.2f}</div>
        <div class="metric-meta">6M Hurdle: +{engine['cash_hurdle_pct']:.2f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Execution Controls
c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([1, 1, 1.4])
with c_ctrl1:
    if st.button(
        "Disable Auto-Execution" if st.session_state.auto_pilot else "Enable Auto-Execution",
        type="primary" if not st.session_state.auto_pilot else "secondary",
        disabled=st.session_state.kill_switch,
        width="stretch",
    ):
        st.session_state.auto_pilot = not st.session_state.auto_pilot
        st.session_state.last_auto_sync = None
        st.rerun()

with c_ctrl2:
    if not st.session_state.kill_switch:
        if st.button("Engage Kill Switch", width="stretch"):
            st.session_state.kill_switch = True
            st.session_state.auto_pilot = False
            st.rerun()
    else:
        if st.button("Release Kill Switch", type="primary", width="stretch"):
            st.session_state.kill_switch = False
            st.rerun()

with c_ctrl3:
    if st.session_state.kill_switch and snap:
        if st.button("Liquidate All Positions & Cancel Open Orders", type="primary", width="stretch"):
            closed = emergency_liquidate_all(cur_user["id"], cur_user["username"], active_key, active_sec, active_paper)
            st.success(f"Liquidated positions: {closed}")
            st.rerun()

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ Navigation Tabs
is_admin = (cur_user["role"] == "admin")
tabs_list = ["Trades", "P&L Graphs", "Trade Logs"]
if is_admin:
    tabs_list.append("Admin & Co-Founder Console")

tabs_rendered = st.tabs(tabs_list)

# ==================================================================
# TAB 1: TRADES
# ==================================================================
with tabs_rendered[0]:
    t_cols = st.columns(len(target_weights) if target_weights else 1)
    for idx, (sym, w) in enumerate(target_weights.items()):
        block_usd = live_capital * w
        base_sym = "QQQ" if sym == "QLD" else ("TLT" if sym == "UBT" else sym)
        ref_px = engine["prices"].get(sym, engine["prices"].get(base_sym, 100.0))
        est_shares = round(block_usd / ref_px, 2) if ref_px > 0 else 0.0
        mom_val = engine["mom_6m"].get(base_sym, 0.0)

        with t_cols[idx]:
            st.markdown(f"""
            <div class="block-card">
                <div class="block-top">
                    <span class="block-sym">{sym}</span>
                    <span class="block-usd">${block_usd:,.2f}</span>
                </div>
                <div class="block-row">
                    <span>Target Allocation</span>
                    <span class="mono">{w*100:.0f}% of Capital</span>
                </div>
                <div class="block-row">
                    <span>Estimated Size</span>
                    <span class="mono">{est_shares:,.2f} Shares @ ${ref_px:,.2f}</span>
                </div>
                <div class="block-row">
                    <span>6M Momentum</span>
                    <span class="mono">{mom_val:+.2f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    if not snap:
        st.info("Add your Alpaca API credentials in the left sidebar to enable live or paper execution.")
    else:
        if snap["open_orders_count"] > 0:
            st.warning(f"{snap['open_orders_count']} pending order(s) queued for market open:")
            st.dataframe(snap["open_orders_df"], width="stretch", hide_index=True)
            if st.button("Cancel Pending Orders", width="stretch"):
                get_alpaca_client(active_key, active_sec, active_paper).cancel_orders()
                st.rerun()

        if st.session_state.auto_pilot and not st.session_state.kill_switch:
            if st.session_state.last_auto_sync != engine["as_of"] and snap["open_orders_count"] == 0:
                auto_orders = execute_big_block_trades(
                    cur_user["id"], cur_user["username"],
                    active_key, active_sec, active_paper, target_weights,
                    min_trade_pct=min_block_pct / 100.0, dry_run=False
                )
                st.session_state.last_auto_sync = engine["as_of"]
                if auto_orders:
                    st.dataframe(pd.DataFrame(auto_orders), width="stretch", hide_index=True)

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button("Preview Rebalance Orders", width="stretch"):
                preview = execute_big_block_trades(
                    cur_user["id"], cur_user["username"],
                    active_key, active_sec, active_paper, target_weights,
                    min_trade_pct=min_block_pct / 100.0, dry_run=True
                )
                if preview:
                    st.dataframe(pd.DataFrame(preview), width="stretch", hide_index=True)
                else:
                    st.success(f"Portfolio is aligned within {min_block_pct}% threshold.")

        with b_col2:
            if st.button("Execute Block Orders", type="primary", disabled=st.session_state.kill_switch, width="stretch"):
                live_orders = execute_big_block_trades(
                    cur_user["id"], cur_user["username"],
                    active_key, active_sec, active_paper, target_weights,
                    min_trade_pct=min_block_pct / 100.0, dry_run=False
                )
                if live_orders:
                    st.dataframe(pd.DataFrame(live_orders), width="stretch", hide_index=True)

        st.markdown("##### Open Positions")
        if len(snap["positions_df"]) > 0:
            st.dataframe(snap["positions_df"], width="stretch", hide_index=True)
        else:
            st.caption("No open positions held.")

# ==================================================================
# TAB 2: P&L GRAPHS
# ==================================================================
with tabs_rendered[1]:
    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("##### Today's P&L")
        intraday_df = fetch_alpaca_intraday_curve(active_key, active_sec, active_paper) if snap else pd.DataFrame()

        if len(intraday_df) > 1:
            fig_today = go.Figure()
            line_col = "#10B981" if intraday_df["Intraday P&L ($)"].iloc[-1] >= 0 else "#EF4444"
            fig_today.add_trace(go.Scatter(
                x=intraday_df["Time"], y=intraday_df["Intraday P&L ($)"],
                mode="lines", name="Today's P&L ($)",
                line=dict(color=line_col, width=2),
                fill="tozeroy", fillcolor="rgba(16, 185, 129, 0.1)" if line_col == "#10B981" else "rgba(239, 68, 68, 0.1)"
            ))
            fig_today.update_layout(
                height=360, paper_bgcolor="#121215", plot_bgcolor="#121215",
                font=dict(family="Inter, sans-serif", color="#A1A1AA"),
                margin=dict(l=12, r=12, t=12, b=12),
                yaxis=dict(gridcolor="#1F1F23", zerolinecolor="#27272A"),
                xaxis=dict(gridcolor="#1F1F23"),
            )
            st.plotly_chart(fig_today, width="stretch")
        else:
            recent_daily_pnl = engine["blended_ret"].iloc[-30:] * live_capital
            fig_daily = go.Figure(go.Bar(
                x=recent_daily_pnl.index, y=recent_daily_pnl.values,
                marker_color=["#10B981" if v >= 0 else "#EF4444" for v in recent_daily_pnl.values]
            ))
            fig_daily.update_layout(
                height=360, paper_bgcolor="#121215", plot_bgcolor="#121215",
                font=dict(family="Inter, sans-serif", color="#A1A1AA"),
                margin=dict(l=12, r=12, t=12, b=12),
                yaxis=dict(title="Daily P&L ($)", gridcolor="#1F1F23"),
                xaxis=dict(title="Last 30 Sessions", gridcolor="#1F1F23"),
            )
            st.plotly_chart(fig_daily, width="stretch")

    with g_col2:
        st.markdown("##### Overall Cumulative P&L")
        horizon = st.radio("Window", options=["1Y", "3Y", "5Y", "ALL (22Y)"], index=3, horizontal=True, label_visibility="collapsed")
        win_map = {"1Y": 252, "3Y": 756, "5Y": 1260, "ALL (22Y)": len(engine["blended_ret"])}
        sub_pnl = ((1.0 + engine["blended_ret"].iloc[-win_map[horizon]:]).cumprod() - 1.0) * live_capital

        fig_overall = go.Figure()
        fig_overall.add_trace(go.Scatter(
            x=sub_pnl.index, y=sub_pnl.values, mode="lines", name="Cumulative P&L",
            line=dict(color="#FAFAFA", width=2),
            fill="tozeroy", fillcolor="rgba(250, 250, 250, 0.06)"
        ))
        fig_overall.update_layout(
            height=325, paper_bgcolor="#121215", plot_bgcolor="#121215",
            font=dict(family="Inter, sans-serif", color="#A1A1AA"),
            margin=dict(l=12, r=12, t=12, b=12),
            yaxis=dict(title="Cumulative Profit ($)", gridcolor="#1F1F23"),
            xaxis=dict(gridcolor="#1F1F23"),
        )
        st.plotly_chart(fig_overall, width="stretch")

# ==================================================================
# TAB 3: TRADE LOGS (PER-USER DATABASE LEDGER + BROKER FILLS)
# ==================================================================
with tabs_rendered[2]:
    st.markdown("##### Your Recorded Platform Trades (Database Ledger)")
    user_db_trades = get_user_recorded_trades(cur_user["id"])
    if len(user_db_trades) > 0:
        st.dataframe(user_db_trades, width="stretch", hide_index=True)
    else:
        st.caption("No platform block orders recorded under your account yet.")

    st.markdown("##### Executed Broker Fills & FIFO Realized P&L (Alpaca)")
    if snap:
        broker_logs = fetch_alpaca_trade_logs(active_key, active_sec, active_paper)
        if len(broker_logs) > 0:
            st.dataframe(broker_logs, width="stretch", hide_index=True)
        else:
            st.caption("No filled orders recorded in this Alpaca account.")
    else:
        st.caption("Connect Alpaca credentials in the sidebar to view live fills.")

    st.markdown("##### Strategy Historical Rebalance Ledger")
    hist_df = engine["historical_logs"].copy()
    if len(hist_df) > 0:
        hist_df["Start Capital ($)"] = (hist_df["start_mult"] * live_capital).round(2)
        hist_df["End Capital ($)"] = (hist_df["end_mult"] * live_capital).round(2)
        hist_df["Realized P&L ($)"] = (hist_df["End Capital ($)"] - hist_df["Start Capital ($)"]).round(2)
        clean_logs = hist_df[["Open Date", "Close Date", "Allocation", "Start Capital ($)", "End Capital ($)", "Return (%)", "Realized P&L ($)"]]
        st.dataframe(clean_logs, width="stretch", hide_index=True)
        st.download_button("Export CSV", data=clean_logs.to_csv(index=False).encode("utf-8"), file_name="amrce_trade_logs.csv", mime="text/csv")

# ==================================================================
# TAB 4: ADMIN & CO-FOUNDER CONSOLE
# ==================================================================
if is_admin:
    with tabs_rendered[3]:
        users_df = get_all_users()
        all_trades_df = get_user_recorded_trades(user_id=None)

        # Admin KPI Overview
        total_users = len(users_df)
        pending_users = int((users_df["is_approved"] == 0).sum())
        vault_linked = int((users_df["has_keys"] == 1).sum())
        platform_vol = float(users_df["total_volume"].sum())

        st.markdown(f"""
        <div class="metric-strip">
            <div class="metric-box">
                <div class="metric-label">Total Registered Users</div>
                <div class="metric-val">{total_users}</div>
                <div class="metric-meta">{pending_users} Pending Approval</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Connected Broker Vaults</div>
                <div class="metric-val">{vault_linked}</div>
                <div class="metric-meta">AES-256 Encrypted</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Total Platform Orders</div>
                <div class="metric-val">{len(all_trades_df)}</div>
                <div class="metric-meta">Logged in SQLite DB</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Cumulative Order Volume</div>
                <div class="metric-val">${platform_vol:,.0f}</div>
                <div class="metric-meta">Across All Users</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        admin_sub1, admin_sub2, admin_sub3 = st.tabs([
            "User & Co-Founder Access Control",
            "Global Platform Trade Ledger",
            "Security & Audit Trail"
        ])

        with admin_sub1:
            col_provision, col_directory = st.columns([1, 2.2])

            with col_provision:
                st.markdown("##### Provision User / Co-Founder")
                with st.form("admin_create_user", border=True):
                    new_u = st.text_input("Username", placeholder="e.g., cofounder_name")
                    new_p = st.text_input("Temporary Password", type="password")
                    new_r = st.selectbox("Role", options=["admin", "user"], format_func=lambda x: "Admin / Co-Founder" if x == "admin" else "Standard User")
                    if st.form_submit_button("Create Approved Account", width="stretch"):
                        ok, msg = create_user_by_admin(cur_user["username"], new_u, new_p, role=new_r, approved=True)
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

            with col_directory:
                st.markdown("##### User Directory & Permissions")
                for _, row in users_df.iterrows():
                    uid = int(row["id"])
                    uname = str(row["username"])
                    urole = str(row["role"])
                    uapp = bool(row["is_approved"])
                    ukeys = "Vault Linked" if row["has_keys"] else "No Keys"
                    utrades = int(row["total_trades"])

                    with st.container(border=True):
                        r1, r2, r3, r4, r5 = st.columns([1.4, 1, 1, 1, 1])
                        r1.markdown(f"**{uname}**  \n`{urole.upper()}` · `{ukeys}`")
                        r2.markdown(f"Status  \n`{'APPROVED' if uapp else 'PENDING'}`")
                        r3.markdown(f"Trades  \n`{utrades} Orders`")

                        # Prevent modifying own active session role/deletion accidentally
                        if uname != cur_user["username"]:
                            with r4:
                                if uapp:
                                    if st.button("Suspend", key=f"rev_{uid}", width="stretch"):
                                        set_user_approval(cur_user["username"], uid, 0)
                                        st.rerun()
                                else:
                                    if st.button("Approve", key=f"app_{uid}", type="primary", width="stretch"):
                                        set_user_approval(cur_user["username"], uid, 1)
                                        st.rerun()
                            with r5:
                                if urole == "user":
                                    if st.button("Make Admin", key=f"prom_{uid}", width="stretch"):
                                        set_user_role(cur_user["username"], uid, "admin")
                                        st.rerun()
                                else:
                                    if st.button("Make User", key=f"dem_{uid}", width="stretch"):
                                        set_user_role(cur_user["username"], uid, "user")
                                        st.rerun()
                        else:
                            r4.caption("Active Session")
                            r5.caption("Founder / Admin")

        with admin_sub2:
            st.markdown("##### All User Executed Trades Across Platform")
            if len(all_trades_df) > 0:
                st.dataframe(all_trades_df, width="stretch", hide_index=True)
            else:
                st.caption("No user trades have been executed yet.")

        with admin_sub3:
            st.markdown("##### Security & Admin Audit Logs")
            audit_df = get_audit_logs()
            if len(audit_df) > 0:
                st.dataframe(audit_df, width="stretch", hide_index=True)
            else:
                st.caption("No audit events logged yet.")