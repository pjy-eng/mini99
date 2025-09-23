# app.py (最终优化版 - 增加7天时间过滤)

import streamlit as st
import requests
from urllib.parse import quote
import datetime  # 1. 导入 datetime 模块

# --- 你的 API 密钥 ---
API_KEY = "45ad095db3c24b7794771093799e01e6"


# --- 核心功能函数 ---
def fetch_and_display_news(search_keyword):
    """根据给定的关键词，获取过去7天的新闻并展示在页面上。"""

    with st.spinner(f"pjy正在搜索关于 '{search_keyword}' 的近期全球新闻..."):

        # --- 2. 新增日期计算逻辑 ---
        # 获取今天的日期
        today = datetime.date.today()
        # 计算7天前的日期
        seven_days_ago = today - datetime.timedelta(days=7)
        # 将日期格式化成 API 需要的 YYYY-MM-DD 格式
        from_date = seven_days_ago.strftime("%Y-%m-%d")

        # 对关键词进行URL编码
        encoded_keyword = quote(search_keyword)

        # --- 3. 在URL中加入 from 参数，并按发布日期排序 ---
        url = (f"https://newsapi.org/v2/everything?"
               f"q={encoded_keyword}&language=en"
               f"&from={from_date}"  # <-- 新增的时间过滤器
               f"&sortBy=publishedAt" # <-- 按最新发布排序
               f"&apiKey={API_KEY}")

        try:
            response = requests.get(url)
            articles = []
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    articles = data.get("articles", [])[:15]
            else:
                st.error(f"API 请求失败，状态码: {response.status_code}, 响应: {response.text}")

        except Exception as e:
            st.error(f"请求API时发生网络错误: {e}")
            articles = []

        if not articles:
            st.warning(f"过去7天未能获取到关于 '{search_keyword}' 的新闻，请稍后再试。")
        else:
            st.success(f"成功获取到 {len(articles)} 条关于 '{search_keyword}' 的近期热门新闻！")

            for i, article in enumerate(articles):
                title = article.get("title", "无标题")
                description = article.get("description", "无简介")
                source_url = article.get("url")
                source_name = article.get("source", {}).get("name")

                st.subheader(f"{i + 1}. {title}")

                with st.expander("pjy为你展开/折叠简介 (原文)"):
                    st.write(description)

                col1, col2 = st.columns([1, 4])
                with col1:
                    st.write(f"**来源:** {source_name}")
                with col2:
                    st.link_button("🔗 阅读原文", source_url)

                st.divider()


# --- 页面基础设置 ---
st.set_page_config(
    page_title="pjy全球娱乐文化基地",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 pjy全球娱乐文化基地")


# --- 创建选项卡 ---
tab_entertainment, tab_museum, tab_gallery, tab_music = st.tabs([
    "🌟 娱乐热点",
    "🏛️ 博物馆",
    "🖼️ 美术馆",
    "🎵 音乐"
])

# --- 为每个选项卡定义内容 ---
with tab_entertainment:
    st.header("🌟 全球娱乐热点新闻")
    fetch_and_display_news("entertainment")

with tab_museum:
    st.header("🏛️ 全球博物馆相关新闻")
    fetch_and_display_news("museum")

with tab_gallery:
    st.header("🖼️ 全球美术馆与展览新闻")
    fetch_and_display_news('"art gallery" OR "art exhibition"')

with tab_music:
    st.header("🎵 全球音乐与演唱会新闻")
    fetch_and_display_news('music OR concert OR festival')
