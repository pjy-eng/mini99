import streamlit as st
import pandas as pd
import numpy as np
import faiss
from text2vec import SentenceModel

# 页面设置
st.set_page_config(page_title="Mini 99 拍照助手", layout="centered")
st.title("📸 富士 Mini 99 拍照助手")
st.markdown("当你要拍摄场景时，我来帮你推荐最合适的参数设置 👇")

# 初始化页面状态
if "page" not in st.session_state:
    st.session_state.page = "input"
if "match_row" not in st.session_state:
    st.session_state.match_row = None

# 加载中文句向量模型
model = SentenceModel("shibing624/text2vec-base-chinese")

# 加载 Excel 数据
@st.cache_data
def load_data():
    df = pd.read_excel('scene.xlsx')
    scenes = df["scene"].tolist()
    embeddings = model.encode(scenes, normalize_embeddings=True)
    return df, scenes, embeddings

df, scenes, scene_embeddings = load_data()

# FAISS 构建索引
index = faiss.IndexFlatIP(scene_embeddings[0].shape[0])
index.add(np.array(scene_embeddings))

# 页面一：输入场景
if st.session_state.page == "input":
    query = st.text_input("📷 请输入拍摄描述场景：", placeholder="例如：凌晨在山顶拍日出")
    if st.button("🔍 搜索") and query:
        query_vec = model.encode([query], normalize_embeddings=True)
        D, I = index.search(np.array(query_vec), k=1)
        match_row = df.iloc[I[0][0]]

        # 保存推荐结果并跳转页面
        st.session_state.match_row = match_row
        st.session_state.page = "result"
        st.rerun()  # ⏩ 立即刷新页面以切换视图

# 页面二：展示结果
elif st.session_state.page == "result":
    match_row = st.session_state.match_row
    st.subheader("🎯 推荐设置")
    st.markdown(f"**匹配场景**：{match_row['scene']}")
    st.markdown(f"**模式**：{match_row['mode']}")
    st.markdown(f"**亮度调节**：{match_row['brightness']}")
    st.markdown(f"**滤镜建议**：{match_row['filter']}")
    st.markdown(f"**是否开启闪光**：{'✅ 是' if match_row['flash'] else '❌ 否'}")
    st.markdown(f"**补充建议**：{match_row['note']}")

    # 返回按钮
    if st.button("🔙 返回"):
        st.session_state.page = "input"
        st.session_state.match_row = None
        st.rerun()  # ⏪ 立即刷新页面回到输入界面
