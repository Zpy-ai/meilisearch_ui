import streamlit as st
from meilisearch import Client
import time
import requests



MEILI_URL = "http://10.8.130.31:7700"
api_key = "aSampleMasterKey"
meili_client = Client(MEILI_URL, api_key)



def get_embedding(query):
    url = "http://10.8.130.31:6008/api/v1/embedding"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-proj-mimouse"
    }
    payload = {
        "texts": [query],
        "model": "bge-m3"
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    # 取第一个 embedding
    return data["data"][0]["embedding"]



def search_meilisearch_hybrid(query, knowledge_base, top_k, semantic_ratio):
    try:
        index = meili_client.index(knowledge_base)
        embedding = get_embedding(query)
        results = index.search(
            query,
            {
                "vector": embedding,
                "hybrid": {
                    "semanticRatio": 1 - semantic_ratio,
                    "embedder": "bge_m3"
                },
                "limit": top_k
            }
        )
        return results.get("hits", []), True
    except Exception as e:
        st.error(f"连接 Meilisearch 失败：{str(e)}")
        return [], False


# 侧边栏配置
with st.sidebar:
    st.header("搜索设置")
    # 知识库选择
    knowledge_base = st.selectbox(
        "知识库", 
        ["movies_vector", "movies1","iresearch_vector"],  # 实际索引名
        help="选择要搜索的知识库"
    )
    # 语义系数滑块
    semantic_ratio = st.slider(
        "SemanticRatio", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.1,
        help="调整语义匹配权重"
    )
    # 返回结果数量
    top_k = st.number_input(
        "返回结果数量(topK)", 
        min_value=1, 
        max_value=100, 
        value=10, 
        step=1,
        help="控制搜索结果条数"
    )
    

    st.markdown("---")
    st.markdown(f"### 当前知识库：{knowledge_base}")
    search_time_placeholder = st.empty()  # 搜索耗时
    result_count_placeholder = st.empty()  # 结果数量

# 主界面
st.markdown("## 🔍 知识库搜索")
search_query = st.text_input("请输入搜索关键词", value="AI", help="支持关键词、短语搜索")
search_btn = st.button("搜索", type="primary")

# 搜索逻辑
if search_btn:
    start_time = time.time()
    
    # 调用 Meilisearch 搜索
    results, success = search_meilisearch_hybrid(search_query, knowledge_base, top_k, semantic_ratio)
    
    # 计算耗时
    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000
    
    # 更新侧边栏状态
    search_time_placeholder.markdown(f"### 搜索耗时：{duration_ms:.2f} ms")
    result_count_placeholder.markdown(f"### 返回结果数：{len(results)} 条")
    
    # 展示搜索结果
    if success and results:
        for i, hit in enumerate(results, start=1):
            st.markdown(f"### {i}. {hit.get('﻿标题','无标题')}")
            st.write(f"🆔 SHA256: {hit.get('id', hit.get('file_sha256', '无'))}")
            st.write(f"📊 行业: {hit.get('标签','无')}")
            st.write(f"📅 发布时间: {hit.get('时间', '无')}")
            st.write(f"🔗 海报: {hit.get('poster', '无')}")
            content = hit.get('content', '') or hit.get('abstract', '')
            st.write(f"📝 摘要: {hit.get('描述','无')}")
            # 关键词数组处理
            keywords = hit.get('keyword', [])
            if isinstance(keywords, list):
                keywords = ', '.join(keywords)
            st.write(f"🔑 关键词: {keywords if keywords else '无'}")
            # PDF 链接
            pdf_link = hit.get('链接', hit.get('pdf_link', '无'))
            if pdf_link:
                st.markdown(f"[📎 PDF链接]({pdf_link})")
            # 文件直链
            file_url = hit.get('file_url', '无')
            if file_url:
                st.markdown(f"[📁 文件下载]({file_url})")
            st.divider()
    elif not results:
        st.info("未找到匹配结果，请尝试其他关键词")



#streamlit run airui_reports.py