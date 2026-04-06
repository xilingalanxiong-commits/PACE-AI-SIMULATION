import streamlit as st
import pandas as pd
from openai import OpenAI
import plotly.express as px
import re, os, httpx, json

# ====== 修复：彻底屏蔽系统代理干扰 ======
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

# ====== 页面配置 ======
st.set_page_config(page_title="PACE Lab AI 实验看板", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #8B0000; color: white; font-weight: bold; }
    .stDownloadButton>button { width: auto; background-color: #f0f2f6; color: #31333F; border: 1px solid #dcdfe6; }
    </style>
    """, unsafe_allow_html=True)


# ====== 1. 密钥管理 ======
def get_local_key():
    if os.path.exists("auth.txt"):
        with open("auth.txt", "r") as f:
            return f.read().strip()
    return ""


with st.sidebar:
    st.title("⚙️ 实验参数配置")
    api_key = st.text_input("中继站 API Key", value=get_local_key(), type="password")
    base_url = st.text_input("中继 Base URL", value="https://api.gptsapi.net/v1")
    model_name = st.selectbox("选择模拟模型", ["gpt-4o-mini", "gpt-4o", "deepseek-r1", "claude-sonnet-4-6", "o3-mini"])
    temp = st.slider("Temperature (决策随机性)", 0.0, 2.0, 1.0, 0.1)
    sample_size = st.number_input("AI 被试样本量 (N)", min_value=1, max_value=100, value=5)

    if st.button("💾 更新本地 Key"):
        with open("auth.txt", "w") as f:
            f.write(api_key)
        st.success("Key 已存至本地")


# ====== 2. 核心逻辑 ======
def get_client():
    clean_http_client = httpx.Client(trust_env=False)
    return OpenAI(api_key=api_key, base_url=base_url, http_client=clean_http_client)


def run_ai_simulation(prompt, n, temperature):
    if not api_key:
        st.error("❌ 错误：请先设置 API Key")
        return []
    client = get_client()
    results = []
    prog_bar = st.progress(0)
    status = st.empty()
    for i in range(n):
        status.markdown(f"🔔 **正在模拟第 {i + 1}/{n} 位被试...**")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=float(temperature)
            )
            results.append(response.choices[0].message.content)
        except Exception as e:
            st.error(f"采样失败: {str(e)}")
        prog_bar.progress((i + 1) / n)
    status.markdown("✅ **实验采样完成！**")
    return results


def dynamic_semantic_analysis(responses):
    if not responses: return None
    client = get_client()
    analysis_prompt = f"""
    分析以下消费者原始回答，归纳出 3-5 个核心语义类别，统计每个类别的人数。
    仅返回 JSON 格式：{{"类别标签": 数量}}
    回答内容：
    {chr(10).join([f"- {r[:200]}" for r in responses])}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a research assistant. Return JSON only."},
                      {"role": "user", "content": analysis_prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except:
        return {"分析未完成": len(responses)}


# ====== 3. 课堂教学界面 ======
# 修改 1：更新标题文案
st.title("🧠 PACE Lab: 消费者行为决策模拟")
st.caption("西南财经大学工商管理学院 | 市场营销系 熊希灵课题组")

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("🛠️ 实验情境设计")
    t1 = st.text_area("1. 被试特征设定", placeholder="例如：你是一个追求性价比、有家庭责任感的西财校友...", height=120)
    t2 = st.text_area("2. 决策任务描述", placeholder="例如：面对某折叠屏手机，你的购买意向是什么？", height=120)

    if st.button("🚀 启动现场模拟实验"):
        if t1 and t2:
            raw_results = run_ai_simulation(f"{t1}\n{t2}", sample_size, temp)
            st.session_state['results'] = raw_results
            with st.spinner("AI 正在总结语义分布..."):
                st.session_state['chart_data'] = dynamic_semantic_analysis(raw_results)
        else:
            st.warning("请完整填写设定")

with right_col:
    st.subheader("📊 模拟观点/决策分布图")
    if 'chart_data' in st.session_state:
        data_map = st.session_state['chart_data']
        df_pie = pd.DataFrame(list(data_map.items()), columns=['语义分类', '被试人数'])

        # 修改 2：更换为高对比度多色系
        fig = px.pie(
            df_pie,
            values='被试人数',
            names='语义分类',
            # 使用 Plotly 标准对比色系：红、黄、蓝、绿等
            color_discrete_sequence=px.colors.qualitative.Bold,
            hole=0.4
        )
        fig.update_traces(textinfo='percent+label', pull=[0.05, 0, 0, 0])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("数据分析中，此处将自动展示被试决策的多样性比例。")

st.divider()

# ====== 4. 数据导出 ======
if 'results' in st.session_state:
    h1, h2 = st.columns([3, 1])
    with h1:
        st.subheader("📄 AI 被试原始数据 (Raw Data)")
    with h2:
        df_csv = pd.DataFrame({"ID": [f"#{i + 1}" for i in range(len(st.session_state['results']))],
                               "Response": st.session_state['results']})
        csv_data = df_csv.to_csv(index=False).encode('utf-8-sig')
        st.download_button(label="📥 下载 CSV 数据", data=csv_data, file_name='PACE_Lab_Results.csv')

    tabs = st.tabs([f"被试 #{i + 1}" for i in range(len(st.session_state['results']))])
    for i, res in enumerate(st.session_state['results']):
        with tabs[i]:
            st.markdown(res)

st.markdown(
    "<br><hr><center style='color: gray; font-size: 0.8em;'>Xiong, Wong, Huang, & Peng（2024）| 消费者行为学实验演示系统</center>",
    unsafe_allow_html=True)