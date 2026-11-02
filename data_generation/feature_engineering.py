"""
feature_engineering.py
=======================
把原始交易/餘額/互動資料轉成可餵給模型的客戶特徵。四組特徵各自對應商業問題
裡的一個輸入訊號:

  1. 閒置資金特徵     <- deposit_balance_monthly
  2. MCC 消費輪廓特徵  <- card_transactions_monthly(冷啟動替代訊號的核心)
  3. App 互動特徵      <- app_engagement_monthly(僅部分客戶有紀錄)
  4. 客戶屬性特徵      <- client_profile

注意:這個模組預期吃進來的是「已清理過」的資料(income 離群值已處理、
重複交易列已去除、缺月資料已標記)。清理流程在 notebook 的「資料清理」
段落示範,不是這個模組的責任——特徵工程與資料清理刻意拆開,是真實資料
pipeline 常見的分工方式。
"""

import numpy as np
import pandas as pd


def build_idle_balance_features(deposit_balance: pd.DataFrame) -> pd.DataFrame:
    """
    產出每位客戶的:近6個月平均餘額、餘額成長趨勢(斜率)、是否符合閒置門檻。
    閒置門檻用全體客戶餘額的分位數決定,而不是寫死金額。
    """
    df = deposit_balance.copy()
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    df = df.sort_values(["customer_id", "month"])

    def _slope(group: pd.DataFrame) -> float:
        if len(group) < 2:
            return 0.0
        x = np.arange(len(group))
        y = group["checking_balance"].values
        return float(np.polyfit(x, y, 1)[0])

    recent = df.groupby("customer_id").tail(6)
    avg_balance_6m = recent.groupby("customer_id")["checking_balance"].mean().rename("avg_balance_6m")
    n_months_observed = df.groupby("customer_id")["month"].nunique().rename("n_balance_months_observed")
    slope = df.groupby("customer_id").apply(_slope).rename("balance_trend_slope")

    out = pd.concat([avg_balance_6m, n_months_observed, slope], axis=1).reset_index()

    idle_threshold = out["avg_balance_6m"].quantile(0.5)
    out["idle_flag"] = out["avg_balance_6m"] >= idle_threshold
    return out


def build_mcc_persona_features(card_transactions: pd.DataFrame) -> pd.DataFrame:
    """
    把每位客戶的 MCC 消費金額轉成「消費類別占比向量」——冷啟動替代訊號的核心
    特徵。同時打上一個簡化的 persona 標籤方便 EDA 講故事用,但實際餵給推薦
    模型的是完整占比向量。
    """
    pivot = (
        card_transactions.groupby(["customer_id", "mcc_category"])["transaction_amount"]
        .sum()
        .unstack(fill_value=0)
    )
    pct = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
    pct.columns = [f"pct_{c}" for c in pct.columns]

    persona_map = {
        "pct_母嬰用品": "穩健家庭型", "pct_旅遊": "生活體驗型", "pct_3C科技": "科技早期採用型",
        "pct_精品/百貨": "高消費品味型", "pct_外送平台": "都會便利型",
    }
    dominant_cat = pivot.idxmax(axis=1)
    dominant_persona = dominant_cat.map(lambda c: persona_map.get(f"pct_{c}", "一般型"))

    out = pct.reset_index()
    out["dominant_persona"] = dominant_persona.values
    out["dominant_mcc_category"] = dominant_cat.values
    out["total_card_spend_6m"] = pivot.sum(axis=1).values
    return out


def build_app_engagement_features(app_engagement: pd.DataFrame, all_customer_ids: pd.Series) -> pd.DataFrame:
    """
    有 App 理財瀏覽紀錄的客戶,算出總點擊數與最常點擊的標籤;沒有紀錄的客戶
    (目標客群主力)則填 0——這個 0 不是缺值,是「冷啟動」這個商業事實本身。
    """
    if len(app_engagement) == 0:
        base = pd.DataFrame({"customer_id": all_customer_ids})
        base["total_app_clicks_6m"] = 0
        base["top_clicked_tag"] = None
        return base

    total_clicks = app_engagement.groupby("customer_id")["click_count"].sum().rename("total_app_clicks_6m")
    tag_pivot = app_engagement.groupby(["customer_id", "tag_clicked"])["click_count"].sum().unstack(fill_value=0)
    top_tag = tag_pivot.idxmax(axis=1).rename("top_clicked_tag") if not tag_pivot.empty else pd.Series(dtype=object, name="top_clicked_tag")

    out = pd.DataFrame({"customer_id": all_customer_ids}).merge(
        total_clicks.reset_index(), on="customer_id", how="left"
    ).merge(top_tag.reset_index(), on="customer_id", how="left")
    out["total_app_clicks_6m"] = out["total_app_clicks_6m"].fillna(0)
    return out


def encode_client_attributes(client_profile: pd.DataFrame) -> pd.DataFrame:
    risk_order = {"保守型": 1, "穩健型": 2, "積極型": 3}
    out = client_profile[[
        "customer_id", "kyc_risk_rating", "age", "tenure_months", "has_children",
        "education_level", "income_tier", "income_annual_wan",
        "app_click_history_flag", "has_wealth_product",
    ]].copy()
    out["risk_tier_encoded"] = out["kyc_risk_rating"].map(risk_order)
    return out


def build_feature_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """整合四組特徵成單一 customer-level 特徵表,供傾向評分模型與推薦模組共用。"""
    idle = build_idle_balance_features(tables["deposit_balance"])
    persona = build_mcc_persona_features(tables["card_transactions"])
    attrs = encode_client_attributes(tables["client_profile"])
    engagement = build_app_engagement_features(
        tables.get("app_engagement", pd.DataFrame(columns=["customer_id", "tag_clicked", "click_count"])),
        tables["client_profile"]["customer_id"],
    )

    feat = (
        attrs
        .merge(idle, on="customer_id", how="left")
        .merge(persona, on="customer_id", how="left")
        .merge(engagement, on="customer_id", how="left")
    )

    feat["cold_start_flag"] = ~feat["app_click_history_flag"]
    return feat
