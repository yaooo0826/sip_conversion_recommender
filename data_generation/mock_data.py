"""
mock_data.py
============
產生正式版模擬資料(v2)。欄位結構與分布依據記錄於 docs/data_dictionary.md,
數值分布已用真實下載的 Kaggle 資料集校準:

  - Credit Card Customers(BankChurners, 10,127筆):Income_Category 真實比例、
    Total_Trans_Amt / Total_Trans_Ct / Credit_Limit 的均值與離散程度
  - Customer Personality Analysis(2,240筆):各消費品項金額的相對占比形狀、
    NumWebVisitsMonth 的均值與標準差、家庭結構(有無子女)欄位

設計原則(對照 Gemini 原始 prompt 的三點優化,詳見 data_dictionary.md):
  1. 五張正規化表(client → balance/transactions/engagement → product),
     不壓成一張寬表,對齊 Berka 資料集示範的真實銀行關聯式結構。
  2. 消費類別 -> KYC屬性是機率性關聯,不是決定性規則。
  3. 刻意注入真實觀察到的髒資料型態(Unknown收入比例、離群值、缺月資料、
     重複列),而不是隨機亂數缺值。
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

MCC_CATEGORIES = [
    "母嬰用品", "3C科技", "旅遊", "外送平台", "超市/量販",
    "餐飲", "精品/百貨", "加油/交通", "居家生活", "其他",
]

RISK_RATINGS = ["保守型", "穩健型", "積極型"]

# 校準自 BankChurners 真實 Income_Category 比例(10,127筆計算得出)
INCOME_TIERS = ["<40萬", "40-60萬", "60-80萬", "80-120萬", "120萬+", "Unknown"]
INCOME_TIER_PROBS = np.array([0.351634, 0.176755, 0.138442, 0.151575, 0.071788, 0.109805])
INCOME_TIER_PROBS = INCOME_TIER_PROBS / INCOME_TIER_PROBS.sum()  # 校正浮點數捨入誤差
INCOME_TIER_MIDPOINT = {  # 新台幣年收入中位數估計(萬元),用於數值化特徵
    "<40萬": 30, "40-60萬": 50, "60-80萬": 70, "80-120萬": 100, "120萬+": 150, "Unknown": np.nan,
}

# 匯率假設(新台幣/美元):BankChurners 的交易金額統計值是美元,產生刷卡資料時換算成新台幣。
# v2 修正:v1 直接把美元數字當成新台幣,導致每月刷卡中位數只有約 NT$355。
USD_TO_TWD = 30

# App 理財標籤,對應消費persona的點擊傾向
APP_TAGS = ["高股息", "美債", "低碳永續", "科技", "平衡型"]


# ----------------------------------------------------------------------------
# 1. 信用卡消費(先產生,因為 KYC 屬性要依消費輪廓機率性推算)
# ----------------------------------------------------------------------------

def _spending_weights_for_customer(age: int, has_children: bool) -> np.ndarray:
    """
    每位客戶的消費類別權重,依人口統計特徵微調——這是 Customer Personality
    資料集裡「家庭結構影響消費品項」邏輯的簡化實作:有子女的家庭在母嬰用品/
    超市權重較高;年輕客群在3C/外送權重較高。權重仍帶隨機性,不是決定性。
    """
    base = RNG.dirichlet(np.ones(len(MCC_CATEGORIES)) * 0.7)
    idx = {c: i for i, c in enumerate(MCC_CATEGORIES)}

    if has_children:
        base[idx["母嬰用品"]] += RNG.uniform(0.15, 0.35)
        base[idx["超市/量販"]] += RNG.uniform(0.05, 0.15)
    if age < 35:
        base[idx["3C科技"]] += RNG.uniform(0.10, 0.25)
        base[idx["外送平台"]] += RNG.uniform(0.05, 0.15)
    if age >= 50:
        base[idx["居家生活"]] += RNG.uniform(0.05, 0.15)
        base[idx["加油/交通"]] += RNG.uniform(0.03, 0.10)

    return base / base.sum()


def generate_card_transactions(client_profile: pd.DataFrame, n_months: int = 6) -> pd.DataFrame:
    """
    交易金額/次數的總量級校準自 BankChurners 真實統計量:
      Total_Trans_Amt  均值 US$4,404 / std US$3,397(年度值,乘以 USD_TO_TWD 換算成新台幣,再換算成月)
      Total_Trans_Ct   均值 64.9 筆(年度值,換算成月)
    """
    rows = []
    months = pd.period_range("2026-03", periods=n_months, freq="M")
    annual_amt_mean, annual_amt_std = 4404 * USD_TO_TWD, 3397 * USD_TO_TWD
    annual_ct_mean = 64.9

    for _, cust in client_profile.iterrows():
        cid, age, has_children = cust["customer_id"], cust["age"], cust["has_children"]
        weights = _spending_weights_for_customer(age, has_children)

        # 個人年度消費總量(右偏,對齊 BankChurners std/mean ≈ 0.77 的離散程度)
        personal_annual_amt = max(500 * USD_TO_TWD, RNG.normal(annual_amt_mean, annual_amt_std))
        monthly_total = personal_annual_amt / 12
        monthly_total_ct = max(2, RNG.normal(annual_ct_mean / 12, 2.0))

        for m in months:
            # 交易「次數」用 multinomial 依權重把整數筆數分配到各類別,總和精準等於
            # monthly_total_ct——不能像原本那樣讓每個有權重的類別各自四捨五入後
            # 再取 max(1, ...),那樣會讓每個「有分到權重」的類別都至少算 1 筆,
            # 類別一多,總筆數就會系統性地被灌水(這是實際發現過的校準錯誤:
            # 平均值從目標 64.9 筆/年膨脹到 106 筆/年,差了 63%,修正後才對齊)。
            n_ct_this_month = max(0, int(round(monthly_total_ct * RNG.normal(1.0, 0.15))))
            ct_allocation = RNG.multinomial(n_ct_this_month, weights) if n_ct_this_month > 0 else np.zeros(len(weights), dtype=int)

            # 金額也改用「當月實際筆數分配」而不是原始權重 w 去分攤——如果金額
            # 還是照 w 算,某類別當月被 multinomial 分到 0 筆卻仍嘗試算金額會被
            # 跳過,等於憑空漏掉那部分金額,總金額會系統性偏低(這裡也是實測抓到
            # 的問題:改完次數的算法後,金額一度從目標 4,404 掉到 2,682)。用「該
            # 類別當月筆數 / 當月總筆數」重新當作金額分攤比例,才能讓總金額跟總
            # 筆數兩個指標同時對得上目標值,而不是顧此失彼。
            if n_ct_this_month > 0:
                amount_shares = ct_allocation / n_ct_this_month
            else:
                amount_shares = np.zeros(len(weights))

            for cat, cat_count, share in zip(MCC_CATEGORIES, ct_allocation, amount_shares):
                if cat_count == 0:
                    continue
                amount = monthly_total * share * RNG.normal(1.0, 0.2)
                if amount <= 0:
                    continue
                rows.append({
                    "customer_id": cid,
                    "transaction_month": str(m),
                    "mcc_category": cat,
                    "transaction_amount": round(amount, 0),
                    "transaction_count": int(cat_count),
                })

    df = pd.DataFrame(rows)

    # --- 刻意注入的髒資料:重複交易列(模擬批次匯入常見錯誤,約0.3%) ---
    n_dupes = int(len(df) * 0.003)
    if n_dupes > 0:
        dupe_rows = df.sample(n=n_dupes, random_state=7)
        df = pd.concat([df, dupe_rows], ignore_index=True)

    return df


def _dominant_category(card_transactions: pd.DataFrame) -> pd.Series:
    pivot = (
        card_transactions.groupby(["customer_id", "mcc_category"])["transaction_amount"]
        .sum()
        .unstack(fill_value=0)
    )
    return pivot.idxmax(axis=1)


def _assign_kyc_rating(dominant_cat: str, age: int, income_midpoint: float) -> str:
    """
    機率性(非決定性)KYC 指派邏輯。
    - 母嬰用品/超市 主導 -> 70% 機率保守/穩健,30% 機率仍可能是積極型
    - 3C科技/外送 主導 + 年齡<35 -> 55% 機率積極型
    - 其餘依收入層級給予基礎機率分布
    這樣 EDA 階段做卡方檢定,關聯性存在但不會趨近完美,才符合真實世界樣貌。
    """
    if dominant_cat in ("母嬰用品", "超市/量販"):
        probs = [0.40, 0.30, 0.30] if age < 35 else [0.45, 0.35, 0.20]
    elif dominant_cat in ("3C科技", "外送平台") and age < 35:
        probs = [0.15, 0.30, 0.55]
    elif pd.notna(income_midpoint) and income_midpoint >= 100:
        probs = [0.20, 0.35, 0.45]
    else:
        probs = [0.45, 0.40, 0.15]
    return RNG.choice(RISK_RATINGS, p=probs)


# ----------------------------------------------------------------------------
# 2. 客戶主檔
# ----------------------------------------------------------------------------

def generate_client_profile_base(n_customers: int) -> pd.DataFrame:
    customer_id = [f"C{100000+i}" for i in range(n_customers)]
    age = RNG.normal(38, 11, n_customers).clip(20, 70).astype(int)
    has_children = RNG.choice([True, False], n_customers, p=[0.5, 0.5])
    education_level = RNG.choice(
        ["高中以下", "大學", "研究所以上"], n_customers, p=[0.25, 0.55, 0.20]
    )
    tenure_months = (RNG.lognormal(mean=3.6, sigma=0.8, size=n_customers)).clip(3, 240).astype(int)

    income_tier = RNG.choice(INCOME_TIERS, n_customers, p=INCOME_TIER_PROBS)
    income_annual_wan = np.array([
        INCOME_TIER_MIDPOINT[t] * RNG.normal(1.0, 0.2) if pd.notna(INCOME_TIER_MIDPOINT[t]) else np.nan
        for t in income_tier
    ])

    has_credit_card = RNG.choice([True, False], n_customers, p=[0.75, 0.25])
    has_deposit_account = np.where(has_credit_card, RNG.choice([True, False], n_customers, p=[0.7, 0.3]), True)
    has_wealth_product = RNG.choice([True, False], n_customers, p=[0.15, 0.85])
    app_click_history_flag = RNG.choice([True, False], n_customers, p=[0.3, 0.7])

    df = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "has_children": has_children,
        "education_level": education_level,
        "tenure_months": tenure_months,
        "income_tier": income_tier,
        "income_annual_wan": income_annual_wan,
        "has_credit_card": has_credit_card,
        "has_deposit_account": has_deposit_account,
        "has_wealth_product": has_wealth_product,
        "app_click_history_flag": app_click_history_flag,
    })

    # --- 刻意注入的髒資料:收入離群值(對照 Customer Personality 真實案例 Income=666,666) ---
    outlier_idx = RNG.choice(df.index, size=max(1, int(n_customers * 0.0015)), replace=False)
    df.loc[outlier_idx, "income_annual_wan"] = 999999  # 明顯異常值,待notebook資料清理階段處理

    return df


# ----------------------------------------------------------------------------
# 3. 活存餘額(與收入/消費呈正相關,但非決定性)
# ----------------------------------------------------------------------------

def generate_deposit_balance(client_profile: pd.DataFrame, card_totals: pd.Series, n_months: int = 12) -> pd.DataFrame:
    rows = []
    months = pd.period_range("2025-08", periods=n_months, freq="M")

    for _, cust in client_profile.iterrows():
        cid = cust["customer_id"]
        income_component = cust["income_annual_wan"] if pd.notna(cust["income_annual_wan"]) and cust["income_annual_wan"] < 500 else 50
        spend_component = card_totals.get(cid, 3000 * USD_TO_TWD)

        # 財富的共同潛在因子:收入 + 刷卡總量 共同驅動活存水準,加上獨立雜訊,
        # 讓「活存與刷卡正相關」但不是完全決定性(對應優化建議2的精神)
        latent_wealth = 0.6 * np.log1p(income_component * 1000) + 0.4 * np.log1p(spend_component) + RNG.normal(0, 0.6)
        base = np.exp(latent_wealth) * RNG.uniform(0.8, 1.2)

        trend = RNG.normal(0, 0.02)
        for i, m in enumerate(months):
            noise = RNG.normal(1.0, 0.05)
            balance = max(0, base * (1 + trend * i) * noise)
            rows.append({"customer_id": cid, "month": str(m), "checking_balance": round(balance, 0)})

    df = pd.DataFrame(rows)

    # --- 刻意注入的髒資料:少數客戶有缺月資料(模擬新開戶/資料同步問題,約2%客戶) ---
    all_customers = client_profile["customer_id"].unique()
    n_missing_customers = int(len(all_customers) * 0.02)
    missing_customers = RNG.choice(all_customers, size=n_missing_customers, replace=False)
    drop_mask = pd.Series(False, index=df.index)
    for cid in missing_customers:
        cust_rows = df.index[df["customer_id"] == cid]
        if len(cust_rows) > 2:
            n_drop = RNG.integers(1, 3)
            drop_idx = RNG.choice(cust_rows, size=n_drop, replace=False)
            drop_mask.loc[drop_idx] = True
    df = df[~drop_mask].reset_index(drop=True)

    return df


# ----------------------------------------------------------------------------
# 4. App 理財瀏覽行為(僅有 app_click_history_flag=True 的客戶有紀錄)
# ----------------------------------------------------------------------------

def generate_app_engagement(client_profile: pd.DataFrame, dominant_cat_map: pd.Series, n_months: int = 6) -> pd.DataFrame:
    """
    整體互動強度校準自 Customer Personality `NumWebVisitsMonth`(均值5.3、std2.4)。
    標籤點擊傾向依消費persona做機率性(非決定性)加權。
    """
    rows = []
    months = pd.period_range("2026-03", periods=n_months, freq="M")
    engaged = client_profile[client_profile["app_click_history_flag"]]

    tag_weight_map = {
        "3C科技": {"科技": 0.5, "高股息": 0.15, "美債": 0.1, "低碳永續": 0.1, "平衡型": 0.15},
        "母嬰用品": {"美債": 0.35, "平衡型": 0.3, "高股息": 0.15, "低碳永續": 0.15, "科技": 0.05},
        "旅遊": {"高股息": 0.3, "平衡型": 0.25, "美債": 0.2, "科技": 0.15, "低碳永續": 0.1},
    }
    default_weights = {t: 1 / len(APP_TAGS) for t in APP_TAGS}

    for _, cust in engaged.iterrows():
        cid = cust["customer_id"]
        dom = dominant_cat_map.get(cid, None)
        weights = tag_weight_map.get(dom, default_weights)
        tags, probs = zip(*weights.items())

        monthly_visits = max(0, RNG.normal(5.3, 2.4))
        for m in months:
            n_clicks_this_month = max(0, int(round(monthly_visits * RNG.normal(1.0, 0.3))))
            if n_clicks_this_month == 0:
                continue
            clicked_tags = RNG.choice(tags, size=n_clicks_this_month, p=probs)
            tag_counts = pd.Series(clicked_tags).value_counts()
            for tag, cnt in tag_counts.items():
                rows.append({"customer_id": cid, "month": str(m), "tag_clicked": tag, "click_count": int(cnt)})

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 5. 商品目錄(不變,虛構名稱,欄位對齊真實 ETF/基金資料集命名)
# ----------------------------------------------------------------------------

def generate_product_catalog() -> pd.DataFrame:
    catalog = [
        {"product_id": "P01", "product_name": "安穩債券精選基金", "asset_class": "債券型", "category": "美債", "risk_rating": 1},
        {"product_id": "P02", "product_name": "全球平衡配置基金", "asset_class": "平衡型", "category": "平衡型", "risk_rating": 2},
        {"product_id": "P03", "product_name": "高股息優選ETF", "asset_class": "股票型", "category": "高股息", "risk_rating": 2},
        {"product_id": "P04", "product_name": "低碳永續主題基金", "asset_class": "股票型", "category": "低碳永續", "risk_rating": 3},
        {"product_id": "P05", "product_name": "科技創新旗艦ETF", "asset_class": "股票型", "category": "科技", "risk_rating": 4},
        {"product_id": "P06", "product_name": "新興市場增長基金", "asset_class": "股票型", "category": "新興市場", "risk_rating": 4},
        {"product_id": "P07", "product_name": "全球旅遊消費主題基金", "asset_class": "股票型", "category": "消費主題", "risk_rating": 3},
        {"product_id": "P08", "product_name": "家庭防禦均衡基金", "asset_class": "股票型", "category": "價值型", "risk_rating": 2},
        # v2 新增:核心資產的典型標的(追蹤大盤指數的 ETF)。不使用亂數,不影響其他資料表
        {"product_id": "P09", "product_name": "全球市值型指數ETF", "asset_class": "股票型", "category": "全球市值型", "risk_rating": 3},
    ]
    return pd.DataFrame(catalog)


# ----------------------------------------------------------------------------
# 6. 轉化標籤(邏輯與 v1 相同精神,依 feature_engineering 產出的特徵表決定)
# ----------------------------------------------------------------------------

def generate_conversion_label(feature_table: pd.DataFrame) -> pd.DataFrame:
    df = feature_table.copy()
    base_rate = 0.05
    logit = np.log(base_rate / (1 - base_rate))

    balance_z = (df["avg_balance_6m"] - df["avg_balance_6m"].mean()) / df["avg_balance_6m"].std()
    slope_z = (df["balance_trend_slope"] - df["balance_trend_slope"].mean()) / df["balance_trend_slope"].std()
    income_z = (df["income_annual_wan"].fillna(df["income_annual_wan"].median())
                - df["income_annual_wan"].mean()) / df["income_annual_wan"].std()

    logit_adj = (
        logit
        + 0.55 * df["idle_flag"].astype(int)
        + 0.35 * slope_z.fillna(0)
        + 0.25 * balance_z.fillna(0)
        + 0.20 * income_z.fillna(0).clip(-3, 3)
        + 0.45 * (df["risk_tier_encoded"] == 3).astype(int)
        + 0.20 * (df["risk_tier_encoded"] == 2).astype(int)
        + 0.30 * (df["tenure_months"] > 60).astype(int)
        - 0.60 * df["cold_start_flag"].astype(int)
        - 0.50 * df["has_wealth_product"].astype(int)
    )
    prob = 1 / (1 + np.exp(-logit_adj))
    converted = RNG.random(len(df)) < prob

    return pd.DataFrame({
        "customer_id": df["customer_id"],
        "converted_to_sip": converted,
    })


# ----------------------------------------------------------------------------
# 組裝流程
# ----------------------------------------------------------------------------

def build_all(n_customers: int = 9000) -> dict[str, pd.DataFrame]:
    import feature_engineering as fe  # 延遲匯入避免循環相依

    client_profile_base = generate_client_profile_base(n_customers)
    card_transactions = generate_card_transactions(client_profile_base)

    dominant_cat_map = _dominant_category(card_transactions)
    card_totals = card_transactions.groupby("customer_id")["transaction_amount"].sum()

    # 依消費輪廓 + 年齡 + 收入,機率性指派 KYC 風險評級
    kyc = []
    for _, cust in client_profile_base.iterrows():
        dom = dominant_cat_map.get(cust["customer_id"], None)
        kyc.append(_assign_kyc_rating(dom, cust["age"], cust["income_annual_wan"]))
    client_profile = client_profile_base.copy()
    client_profile["kyc_risk_rating"] = kyc

    deposit_balance = generate_deposit_balance(client_profile, card_totals)
    app_engagement = generate_app_engagement(client_profile, dominant_cat_map)
    product_catalog = generate_product_catalog()

    partial_tables = {
        "client_profile": client_profile,
        "deposit_balance": deposit_balance,
        "card_transactions": card_transactions,
        "app_engagement": app_engagement,
    }
    feature_table = fe.build_feature_table(partial_tables)
    conversion_label = generate_conversion_label(feature_table)

    return {
        "client_profile": client_profile,
        "deposit_balance": deposit_balance,
        "card_transactions": card_transactions,
        "app_engagement": app_engagement,
        "product_catalog": product_catalog,
        "conversion_label": conversion_label,
    }


# ----------------------------------------------------------------------------
# 7. 進階髒資料注入(export階段才套用,不影響 build_all() 內部特徵工程/建模流程)
# ----------------------------------------------------------------------------
#
# 為什麼分開處理:這裡有幾種髒資料(數值欄位混進文字、日期格式不一致)如果在
# build_all() 內部就注入,會讓 feature_engineering / propensity_model 這些
# 「架構示範」用的模組直接崩潰(因為它們預期拿到乾淨的型態)。所以採用分層設計:
# build_all() 產出的是「輕度髒」但下游模組跑得動的資料(既有的4種問題:收入
# 未揭露/離群值、重複交易列、缺月資料);inject_advanced_messiness() 是在那
# 之上,對「要匯出給使用者自己練習清理」的版本額外疊加更多真實世界常見、且更
# 具破壞性的髒資料型態,模擬多來源系統整合、人工登打錯誤等常見情境。

def inject_advanced_messiness(tables: dict[str, pd.DataFrame], seed: int = 99) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out = {k: v.copy() for k, v in tables.items()}

    # --- client_profile ---
    cp = out["client_profile"]

    # 1) 類別欄位標籤不一致(同一個意思,不同寫法/大小寫/簡繁/中英混雜)
    edu_variants = {
        "高中以下": ["高中以下", "高中(含)以下", "High School", "高中以下 "],
        "大學": ["大學", "大学", "Bachelor", "大學 "],
        "研究所以上": ["研究所以上", "碩士以上", "Graduate", "研究所以上 "],
    }
    edu_dirty_idx = rng.choice(cp.index, size=int(len(cp) * 0.07), replace=False)
    for idx in edu_dirty_idx:
        original = cp.loc[idx, "education_level"]
        variants = edu_variants.get(original)
        if variants:
            cp.loc[idx, "education_level"] = rng.choice(variants)

    kyc_variants = {
        "保守型": ["保守型", "保守", "Conservative", "保守型 "],
        "穩健型": ["穩健型", "穩健", "Moderate", "穩健型 "],
        "積極型": ["積極型", "積極", "Aggressive", "積極型 "],
    }
    kyc_dirty_idx = rng.choice(cp.index, size=int(len(cp) * 0.06), replace=False)
    for idx in kyc_dirty_idx:
        original = cp.loc[idx, "kyc_risk_rating"]
        variants = kyc_variants.get(original)
        if variants:
            cp.loc[idx, "kyc_risk_rating"] = rng.choice(variants)

    # 2) 明顯無效的數值(年齡登打錯誤:0歲、負數、超齡)
    n_invalid_age = max(8, int(len(cp) * 0.004))
    invalid_age_idx = rng.choice(cp.index, size=n_invalid_age, replace=False)
    cp.loc[invalid_age_idx, "age"] = rng.choice([0, -3, 150, 999], size=n_invalid_age)

    # 3) 隨機缺值散落在往來月數欄位(不像收入缺值那樣集中、可解釋,純粹隨機)
    missing_tenure_idx = rng.choice(cp.index, size=int(len(cp) * 0.05), replace=False)
    cp.loc[missing_tenure_idx, "tenure_months"] = np.nan

    # 4) 重複客戶紀錄(同一人因重複建檔被輸入兩次,兩筆資料略有出入)
    dup_customers = cp.sample(n=max(1, int(len(cp) * 0.005)), random_state=seed).copy()
    dup_customers["age"] = dup_customers["age"] + rng.integers(-1, 2, size=len(dup_customers))
    cp = pd.concat([cp, dup_customers], ignore_index=True)

    out["client_profile"] = cp

    # --- card_transactions ---
    ct = out["card_transactions"]

    # 5) MCC 類別標籤不一致
    mcc_variants = {
        "3C科技": ["3C科技", "3c科技", "3C 科技"],
        "母嬰用品": ["母嬰用品", "母嬰用品 ", "母婴用品"],
        "旅遊": ["旅遊", "旅遊 ", "旅游"],
    }
    mcc_dirty_idx = rng.choice(ct.index, size=int(len(ct) * 0.06), replace=False)
    for idx in mcc_dirty_idx:
        original = ct.loc[idx, "mcc_category"]
        variants = mcc_variants.get(original)
        if variants:
            ct.loc[idx, "mcc_category"] = rng.choice(variants)

    # 6) 數值欄位混進文字(系統轉換失敗/人工填寫時常見的哨兵值)
    sentinel_idx = rng.choice(ct.index, size=int(len(ct) * 0.035), replace=False)
    ct["transaction_amount"] = ct["transaction_amount"].astype(object)
    ct.loc[sentinel_idx, "transaction_amount"] = rng.choice(["N/A", "-", "unknown"], size=len(sentinel_idx))

    # 7) 日期格式不一致(模擬多來源系統合併,日期慣例不同)
    date_fmt_idx = rng.choice(ct.index, size=int(len(ct) * 0.06), replace=False)
    ct.loc[date_fmt_idx, "transaction_month"] = ct.loc[date_fmt_idx, "transaction_month"].str.replace("-", "/", regex=False)

    # 8) 交易明細裡出現客戶主檔查無此人的 customer_id(參照完整性問題)
    n_orphans = max(20, int(len(ct) * 0.0003))
    orphan_rows = ct.sample(n=n_orphans, random_state=seed).copy()
    orphan_rows["customer_id"] = [f"C999{900+i}" for i in range(n_orphans)]
    ct = pd.concat([ct, orphan_rows], ignore_index=True)

    out["card_transactions"] = ct

    # --- deposit_balance ---
    db = out["deposit_balance"]

    date_fmt_idx = rng.choice(db.index, size=int(len(db) * 0.05), replace=False)
    db.loc[date_fmt_idx, "month"] = db.loc[date_fmt_idx, "month"].str.replace("-", "/", regex=False)

    n_negative = max(12, int(len(db) * 0.002))
    negative_idx = rng.choice(db.index, size=n_negative, replace=False)
    db.loc[negative_idx, "checking_balance"] = -db.loc[negative_idx, "checking_balance"].abs()

    out["deposit_balance"] = db

    # --- app_engagement ---
    ae = out["app_engagement"]
    if len(ae) > 0:
        tag_variants = {
            "科技": ["科技", "科技 ", "Tech"],
            "美債": ["美債", "美债", "US Bond"],
        }
        tag_dirty_idx = rng.choice(ae.index, size=int(len(ae) * 0.05), replace=False)
        for idx in tag_dirty_idx:
            original = ae.loc[idx, "tag_clicked"]
            variants = tag_variants.get(original)
            if variants:
                ae.loc[idx, "tag_clicked"] = rng.choice(variants)
        out["app_engagement"] = ae

    return out


if __name__ == "__main__":
    tables = build_all(2000)
    for name, df in tables.items():
        print(name, df.shape)
