"""為 conversion_label.csv 加上 conversion_month，不改動任何其他資料表。
設計：
- 特徵觀察期截止 2026-08；標籤期 = 2026-09 ~ 2026-11（觀察期結束後 3 個月內）
- converted_to_sip 的 True/False 完全保留原值
- True 才有月份，False 為空值；格式 YYYY/MM/DD，日固定為 01（代表該月）
- 月份分布逐月遞減（行銷/需求的自然衰退）：45% / 32% / 23%
- 固定亂數種子，可重現
"""
import numpy as np, pandas as pd, sys
SEED = 42
src, dst = sys.argv[1], sys.argv[2]
lab = pd.read_csv(src, encoding="utf-8-sig")
rng = np.random.default_rng(SEED)
months = np.array(["2026/09/01", "2026/10/01", "2026/11/01"])  # 西元年/月/日，以月初代表該月
probs = [0.45, 0.32, 0.23]
lab["conversion_month"] = pd.NA
mask = lab["converted_to_sip"].astype(str).str.strip().str.lower() == "true"
lab.loc[mask, "conversion_month"] = rng.choice(months, size=mask.sum(), p=probs)
lab.to_csv(dst, index=False, encoding="utf-8-sig")
