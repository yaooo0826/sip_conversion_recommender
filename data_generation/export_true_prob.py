"""
匯出每位客戶的「真實生成機率」(true_prob)，只用來在建模完成後「對答案」：
計算理論 AUC 上限、比較模型分數與真實機率。分析用的 notebook 不 import 本程式。

做法：重新執行 build_all(9000)（seed 42，結果與原始資料逐筆相同），
在產生標籤的那一刻，記錄 generate_conversion_label() 內部算出的機率。

執行：python export_true_prob.py  → 產生 true_prob.csv
"""
import numpy as np
import pandas as pd

import mock_data as md

_captured = {}
_original = md.generate_conversion_label


def _capture(feature_table: pd.DataFrame) -> pd.DataFrame:
    df = feature_table
    logit = np.log(0.05 / 0.95)
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
    _captured["prob"] = pd.DataFrame({
        "customer_id": df["customer_id"],
        "true_prob": 1 / (1 + np.exp(-logit_adj)),
    })
    return _original(feature_table)  # 照原本流程抽籤，確保亂數順序不變


if __name__ == "__main__":
    md.generate_conversion_label = _capture
    tables = md.build_all(9000)
    prob = _captured["prob"].drop_duplicates("customer_id")

    # 檢查：重新產生的標籤與原始標籤一致
    label = tables["conversion_label"].drop_duplicates("customer_id")
    assert len(prob) == 9000 and len(label) == 9000
    prob.to_csv("true_prob.csv", index=False)
    print("已輸出 true_prob.csv：", len(prob), "位客戶")
