import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv("kicksharing.csv")
    df["book_start_dttm"] = pd.to_datetime(df["book_start_dttm"], errors="coerce")
    df["book_end_dttm"] = pd.to_datetime(df["book_end_dttm"], errors="coerce")
    df["duration_min"] = (df["book_end_dttm"] - df["book_start_dttm"]).dt.total_seconds()/60
    df["start_hour"] = df["book_start_dttm"].dt.hour
    df["start_month"] = df["book_start_dttm"].dt.month
    return df

df = load_data()

st.title("📊 Дашборд: анализ поездок на самокатах")

st.sidebar.header("Фильтры")
gender = st.sidebar.multiselect(
    "Пол клиента", options=df["gender_cd"].dropna().unique(), default=df["gender_cd"].dropna().unique()
)
month = st.sidebar.multiselect(
    "Месяц поездки", options=sorted(df["start_month"].dropna().unique()), default=sorted(df["start_month"].dropna().unique())
)

filtered_df = df[df["gender_cd"].isin(gender) & df["start_month"].isin(month)]

st.write(f"Количество поездок после фильтрации: {len(filtered_df)}")

st.subheader("Количество пропусков")
na_counts = filtered_df.isna().sum()
st.bar_chart(na_counts)

st.subheader("Потенциальные выбросы (1.5 IQR)")
num_cols = filtered_df.select_dtypes(include=[np.number]).columns.drop(["order_rk","party_rk"], errors="ignore")
outlier_counts = {}
for col in num_cols:
    q1 = filtered_df[col].quantile(0.25)
    q3 = filtered_df[col].quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
    outlier_counts[col] = ((filtered_df[col] < low) | (filtered_df[col] > high)).sum()
st.bar_chart(pd.Series(outlier_counts))

st.subheader("Корреляция между числовыми признаками")
corr = filtered_df[num_cols].corr()
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.subheader("Распределение длительности поездки (0–100 мин)")
fig, ax = plt.subplots(figsize=(8,4))
dur = filtered_df["duration_min"].dropna()
ax.hist(dur[(dur>=0)&(dur<=100)], bins=np.arange(0,105,5))
ax.set_xlabel("Минуты")
ax.set_ylabel("Количество поездок")
st.pyplot(fig)

st.subheader("Количество поездок по часам")
st.bar_chart(filtered_df["start_hour"].value_counts().sort_index())

st.subheader("Количество поездок по месяцам")
st.bar_chart(filtered_df["start_month"].value_counts().sort_index())

st.subheader("Распределение поездок по полу")
st.bar_chart(filtered_df["gender_cd"].value_counts())

st.subheader("Средняя цена за минуту в зависимости от часа старта")
tmp = filtered_df.dropna(subset=["start_hour","nominal_price_rub_amt","duration_min"])
tmp = tmp[tmp["duration_min"] > 0]
tmp["price_per_minute_fact"] = tmp["nominal_price_rub_amt"]/tmp["duration_min"]
mean_by_hour = tmp.groupby("start_hour")["price_per_minute_fact"].mean()
st.line_chart(mean_by_hour)
