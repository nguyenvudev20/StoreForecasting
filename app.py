import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    validate_columns,
    preprocess_training_data,
    train_valid_split_by_date,
    fit_model,
    evaluate_model,
    get_series,
    recursive_forecast,
    compute_inventory_suggestion,
    model_exists,
    load_artifacts,
    save_artifacts,
    FEATURE_COLS,
)

st.set_page_config(
    page_title="Demand Forecasting System",
    page_icon="📦",
    layout="wide"
)

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
.kpi-card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
    padding: 18px 20px;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.15);
}
.kpi-title {
    font-size: 14px;
    color: #cbd5e1;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 28px;
    font-weight: 700;
}
.small-note {
    color: #64748b;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


def kpi_card(title, value):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    return df


st.sidebar.title("📌 Menu")
menu = st.sidebar.radio(
    "Chọn chức năng",
    ["Tổng quan", "Huấn luyện mô hình", "Dashboard", "Dự báo", "Tối ưu tồn kho"]
)

valid_days = st.sidebar.slider("Số ngày validation", 30, 180, 90, 10)
forecast_days = st.sidebar.slider("Số ngày dự báo", 7, 90, 30, 1)
service_z = st.sidebar.selectbox("Mức phục vụ (Z-score)", [1.28, 1.65, 1.96, 2.33], index=1)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"],
    help="Yêu cầu cột: date, store, item, sales"
)

# Auto load model nếu đã có
if "model" not in st.session_state and model_exists():
    model, feature_cols, metrics, valid_result = load_artifacts()
    st.session_state["model"] = model
    st.session_state["feature_cols"] = feature_cols
    st.session_state["metrics"] = metrics
    st.session_state["valid_result"] = valid_result
    st.sidebar.success("Đã load model đã lưu.")

if uploaded_file is None:
    st.info("Hãy upload file CSV từ sidebar để bắt đầu.")
    st.stop()

raw_df = load_csv(uploaded_file)
missing = validate_columns(raw_df)

if missing:
    st.error(f"Thiếu cột bắt buộc: {missing}")
    st.stop()

raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")
raw_df = raw_df.dropna(subset=["date"]).copy()
raw_df = raw_df.sort_values(["store", "item", "date"]).reset_index(drop=True)

# Tổng quan
if menu == "Tổng quan":
    st.title("📦 Hệ thống dự báo nhu cầu và tối ưu tồn kho")
    st.caption("Ứng dụng AI hỗ trợ doanh nghiệp dự báo nhu cầu bán hàng theo cửa hàng và sản phẩm.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Số dòng dữ liệu", f"{len(raw_df):,}")
    with c2:
        kpi_card("Số cửa hàng", raw_df["store"].nunique())
    with c3:
        kpi_card("Số sản phẩm", raw_df["item"].nunique())
    with c4:
        kpi_card("Khoảng ngày", f"{raw_df['date'].min().date()} → {raw_df['date'].max().date()}")

    st.markdown("### Xem dữ liệu")
    st.dataframe(raw_df.head(20), use_container_width=True)

    sales_by_date = raw_df.groupby("date", as_index=False)["sales"].sum()
    fig = px.line(sales_by_date, x="date", y="sales", title="Tổng doanh số theo thời gian")
    st.plotly_chart(fig, use_container_width=True)

# Huấn luyện mô hình
elif menu == "Huấn luyện mô hình":
    st.title("🧠 Huấn luyện mô hình")

    if model_exists():
        st.success("Model đã tồn tại. App sẽ dùng model đã lưu.")
        button_text = "Train lại model"
    else:
        st.warning("Chưa có model. Cần train lần đầu.")
        button_text = "Train model lần đầu"

    if st.button(button_text):
        with st.spinner("Đang xử lý dữ liệu và huấn luyện model..."):
            df_feat = preprocess_training_data(raw_df)
            train_df, valid_df, split_date = train_valid_split_by_date(df_feat, valid_days=valid_days)

            if len(train_df) == 0 or len(valid_df) == 0:
                st.error("Không đủ dữ liệu để huấn luyện.")
                st.stop()

            model = fit_model(train_df)
            metrics, valid_result = evaluate_model(model, valid_df)

            save_artifacts(
                model=model,
                feature_cols=FEATURE_COLS,
                metrics=metrics,
                valid_result=valid_result
            )

            st.session_state["model"] = model
            st.session_state["feature_cols"] = FEATURE_COLS
            st.session_state["metrics"] = metrics
            st.session_state["valid_result"] = valid_result
            st.session_state["split_date"] = split_date

            st.success("Huấn luyện và lưu model thành công.")

    if "metrics" in st.session_state and st.session_state["metrics"] is not None:
        metrics = st.session_state["metrics"]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("MAE", f"{metrics['MAE']:.2f}")
        c2.metric("RMSE", f"{metrics['RMSE']:.2f}")
        c3.metric("R²", f"{metrics['R2']:.4f}")
        c4.metric("MAPE", f"{metrics['MAPE']:.2f}%")
        c5.metric("SMAPE", f"{metrics['SMAPE']:.2f}%")

        st.markdown(
            '<div class="small-note">Với forecasting, nên ưu tiên RMSE, MAE, SMAPE hơn "accuracy".</div>',
            unsafe_allow_html=True
        )

# Dashboard
elif menu == "Dashboard":
    st.title("📊 Dashboard")

    if "model" not in st.session_state:
        st.warning("Hãy vào mục 'Huấn luyện mô hình' để train hoặc load model trước.")
        st.stop()

    selected_store = st.selectbox("Chọn Store", sorted(raw_df["store"].unique().tolist()))
    item_options = sorted(raw_df[raw_df["store"] == selected_store]["item"].unique().tolist())
    selected_item = st.selectbox("Chọn Item", item_options)

    series_df = get_series(raw_df, selected_store, selected_item)
    valid_result = st.session_state.get("valid_result")

    tab1, tab2, tab3 = st.tabs(["Lịch sử bán hàng", "Actual vs Predicted", "Phân tích"])

    with tab1:
        fig_hist = px.line(
            series_df,
            x="date",
            y="sales",
            title=f"Sales History - Store {selected_store}, Item {selected_item}"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        if valid_result is not None:
            vr = valid_result[
                (valid_result["store"] == selected_store) &
                (valid_result["item"] == selected_item)
            ].copy()

            if len(vr) > 0:
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Scatter(x=vr["date"], y=vr["sales"], mode="lines", name="Actual"))
                fig_compare.add_trace(go.Scatter(x=vr["date"], y=vr["pred"], mode="lines", name="Predicted"))
                fig_compare.update_layout(title="Actual vs Predicted")
                st.plotly_chart(fig_compare, use_container_width=True)
            else:
                st.info("Không có dữ liệu validation cho store-item này.")
        else:
            st.info("Chưa có valid_result. Hãy train model ít nhất 1 lần.")

    with tab3:
        month_df = series_df.copy()
        month_df["month"] = month_df["date"].dt.month
        month_avg = month_df.groupby("month", as_index=False)["sales"].mean()

        dow_df = series_df.copy()
        dow_df["dow"] = series_df["date"].dt.day_name()
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_df["dow"] = pd.Categorical(dow_df["dow"], categories=dow_order, ordered=True)
        dow_avg = dow_df.groupby("dow", as_index=False)["sales"].mean()

        a, b = st.columns(2)
        with a:
            st.plotly_chart(
                px.bar(month_avg, x="month", y="sales", title="Average Sales by Month"),
                use_container_width=True
            )
        with b:
            st.plotly_chart(
                px.bar(dow_avg, x="dow", y="sales", title="Average Sales by Day of Week"),
                use_container_width=True
            )

# Dự báo
elif menu == "Dự báo":
    st.title("🔮 Dự báo nhu cầu")

    if "model" not in st.session_state:
        st.warning("Hãy train hoặc load model trước.")
        st.stop()

    model = st.session_state["model"]
    feature_cols = st.session_state.get("feature_cols", FEATURE_COLS)

    selected_store = st.selectbox("Store", sorted(raw_df["store"].unique().tolist()))
    item_options = sorted(raw_df[raw_df["store"] == selected_store]["item"].unique().tolist())
    selected_item = st.selectbox("Item", item_options)

    series_df = get_series(raw_df, selected_store, selected_item)

    if st.button(f"Dự báo {forecast_days} ngày tới"):
        try:
            future_df = recursive_forecast(
                history_df=series_df[["date", "sales"]],
                model=model,
                forecast_days=forecast_days,
                store=selected_store,
                item=selected_item,
                feature_cols=feature_cols
            )

            st.session_state["future_df"] = future_df
            st.session_state["selected_store"] = selected_store
            st.session_state["selected_item"] = selected_item

            st.dataframe(future_df, use_container_width=True)

            fig = go.Figure()
            hist_recent = series_df.tail(90)
            fig.add_trace(go.Scatter(x=hist_recent["date"], y=hist_recent["sales"], mode="lines", name="Historical"))
            fig.add_trace(go.Scatter(x=future_df["date"], y=future_df["forecast_sales"], mode="lines+markers", name="Forecast"))
            fig.update_layout(title=f"Forecast {forecast_days} days - Store {selected_store} / Item {selected_item}")
            st.plotly_chart(fig, use_container_width=True)

            csv_buffer = io.StringIO()
            future_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download forecast CSV",
                data=csv_buffer.getvalue(),
                file_name=f"forecast_store_{selected_store}_item_{selected_item}.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Lỗi dự báo: {e}")

# Tối ưu tồn kho
elif menu == "Tối ưu tồn kho":
    st.title("📦 Tối ưu tồn kho")

    selected_store = st.selectbox("Store", sorted(raw_df["store"].unique().tolist()), key="inv_store")
    item_options = sorted(raw_df[raw_df["store"] == selected_store]["item"].unique().tolist())
    selected_item = st.selectbox("Item", item_options, key="inv_item")

    series_df = get_series(raw_df, selected_store, selected_item)

    current_inventory = st.number_input("Tồn kho hiện tại", min_value=0.0, value=100.0)
    lead_time_days = st.number_input("Lead time (ngày)", min_value=1, value=7)
    horizon = st.number_input("Số ngày nhu cầu cần tính", min_value=1, max_value=90, value=7)

    if (
        "future_df" in st.session_state and
        st.session_state.get("selected_store") == selected_store and
        st.session_state.get("selected_item") == selected_item
    ):
        future_df = st.session_state["future_df"]
        demand_for_horizon = future_df.head(horizon)["forecast_sales"].sum()
        avg_daily_demand = future_df.head(horizon)["forecast_sales"].mean()
    else:
        recent_sales = series_df.tail(max(30, horizon))
        demand_for_horizon = recent_sales.tail(horizon)["sales"].sum()
        avg_daily_demand = recent_sales.tail(horizon)["sales"].mean()

    demand_std = series_df.tail(30)["sales"].std()
    if pd.isna(demand_std):
        demand_std = 0.0

    result = compute_inventory_suggestion(
        current_inventory=current_inventory,
        forecast_demand=demand_for_horizon,
        lead_time_days=int(lead_time_days),
        avg_daily_demand=float(avg_daily_demand),
        demand_std=float(demand_std),
        service_z=float(service_z)
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Safety Stock", f"{result['safety_stock']:.2f}")
    c2.metric("Reorder Point", f"{result['reorder_point']:.2f}")
    c3.metric("Suggested Order", f"{result['suggested_order']:.2f}")

    if current_inventory <= result["reorder_point"]:
        st.warning("Tồn kho hiện tại đang thấp hơn hoặc bằng Reorder Point.")
    else:
        st.success("Tồn kho hiện tại đang an toàn.")