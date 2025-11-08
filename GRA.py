import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

model_name='GRA'
def GRA(df, target_col, rho=0.5):
    # 只保留數值欄位 + 確保 target 還在
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if target_col not in numeric_df.columns:
        raise ValueError(f"target_col '{target_col}' 不在數值欄位內，請確認資料前處理。")

    # 常數欄處理：避免除以 0
    col_max = numeric_df.max()
    col_min = numeric_df.min()
    den = col_max - col_min

    # 建議：剔除常數欄（除了 target 之外）
    constant_cols = den[den == 0].index.tolist()
    constant_cols = [c for c in constant_cols if c != target_col]
    if constant_cols:
        numeric_df = numeric_df.drop(columns=constant_cols)
        # 重新計算 max/min/den
        col_max = numeric_df.max()
        col_min = numeric_df.min()
        den = col_max - col_min

    # 成本型正規化 (越小越好)：(max - x) / (max - min)
    data = (col_max - numeric_df) / den
    # 防呆：若仍有 0 分母（只剩 target 為常數），把結果設為 0
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    target = data[target_col].to_numpy(dtype=float)
    grg = {}

    for col in data.columns:
        if col == target_col:
            continue
        seq = data[col].to_numpy(dtype=float)
        diff = np.abs(target - seq)

        # 用 nan-safe 的 min/max，避免被 NaN 影響
        min_diff = np.nanmin(diff)
        max_diff = np.nanmax(diff)

        # 若完全重合（max_diff=0），該欄 GRG=1
        if max_diff == 0 or np.isclose(max_diff, 0):
            grc = np.ones_like(diff, dtype=float)
        else:
            grc = (min_diff + rho * max_diff) / (diff + rho * max_diff)

        grg[col] = float(np.nanmean(grc))

    return pd.DataFrame(
        sorted(grg.items(), key=lambda x: x[1], reverse=True),
        columns=['Feature', 'Gray Relational Grade']
    )


df=pd.read_csv('dataset\\OK\\All.csv')
df["EUA_Open_t+1"] = df["EUA_Open"].shift(-1)
df = df.drop(columns=["EUA_Open"])
df = df.dropna().reset_index(drop=True)

if "Date" in df.columns:
    df = df.drop(columns=["Date"])
target_col="EUA_Open_t+1"

gra_results=GRA(df,target_col=target_col)
print(gra_results)

output_dir='./finish'
base_dir = os.path.join(output_dir, model_name)
os.makedirs(base_dir,exist_ok=True)

csv_path=os.path.join(base_dir,"gra_results.csv")
gra_results.to_csv(csv_path,index=False,encoding='utf-8-sig')
print(f"csv_finish:{csv_path}")

labels=gra_results['Feature'].tolist()
values=gra_results['Gray Relational Grade'].tolist()

N=len(labels)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles.append(angles[0])
values = values + [values[0]] 

threshold_val = 0.5
threshold = [threshold_val] * N + [threshold_val]

fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
ax.plot(angles, values, "o-", linewidth=2, label="Grey relation degree", color="black")
ax.fill(angles, values, alpha=0.25, color="red")
ax.plot(angles, threshold, "d--", linewidth=1.5, label="Strong relation threshold", color="blue")
# 標籤
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=9, rotation=20)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)

texts = []
for angle, value, label in zip(angles, values, labels + [labels[0]]):
    texts.append(ax.text(angle, value + 0.05, f"{value:.3f}", 
                         ha="center", va="center", fontsize=8,fontweight="bold", color="indianred"))

adjust_text(texts, ax=ax, only_move={'points':'y', 'text':'y'}, 
            arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))

ax.legend(loc='upper right',bbox_to_anchor=(1.2,1.1))
plt.title(f'Gray Relational Analysis (GRA) Result',fontsize=14, pad=40, fontweight="bold")

png_path=os.path.join(base_dir,'gra_results.png')
plt.savefig(png_path,dpi=300,bbox_inches='tight')
plt.close()
print(f"finish:{png_path}")