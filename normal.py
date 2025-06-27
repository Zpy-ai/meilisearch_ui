import streamlit as st
from meilisearch import Client
import time
import requests

from openai import OpenAI


client = OpenAI(
    api_key="sk-139a40229c0e4bd58191a7a2f8c9c8f3",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 初始化 Meilisearch 客户端（替换为你的实际地址，若有 Master Key 需添加）
MEILI_URL = "http://10.8.130.32:7700"
api_key = "ff962c139a9c43142b122d00f8c99f1d"  # 如果有设置 Master Key，请取消注释并填写
meili_client = Client(MEILI_URL, api_key)  # Client(MEILI_URL, "Key")



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
                    "semanticRatio": semantic_ratio,
                    "embedder": "bge_m3"
                },
                "limit": top_k
            }
        )
        return results.get("hits", []), True
    except Exception as e:
        st.error(f"连接 Meilisearch 失败：{str(e)}")
        return [], False

def get_summary_qianwen(text):
    prompt = f"请用中文对以下内容生成简明摘要：\n{text}"
    try:
        response = client.chat.completions.create(
            model="qwen-plus",  # 或你实际支持的模型名
            messages=[
                {"role": "system", "content": "你是一个专业的中文摘要助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=128
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"摘要生成失败: {e}"

# 侧边栏配置
with st.sidebar:
    st.header("搜索设置")
    # 知识库选择（需与 Meilisearch 中的索引名一致）
    knowledge_base = st.selectbox(
        "知识库", 
        ["broker_reports"],  # 替换为实际索引名
        help="选择要搜索的知识库"
    )
    # 语义系数滑块（可根据业务逻辑传递给 Meilisearch）
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
    
    # 状态显示（搜索后动态更新）
    st.markdown("---")
    st.markdown(f"### 当前知识库：{knowledge_base}")
    search_time_placeholder = st.empty()  # 搜索耗时
    result_count_placeholder = st.empty()  # 结果数量

# 主界面布局
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
            st.markdown(f"### {i}. {hit.get('title', '无标题')}")
            st.write(f"🆔 SHA256: {hit.get('_sha256', hit.get('file_sha256', '无'))}")
            st.write(f"👤 作者: {hit.get('author', '无')}")
            st.write(f"🏢 机构: {hit.get('organization', '无')}")
            st.write(f"📊 行业: {hit.get('industry', '无')}")
            #st.write(f"📝 摘要: {hit.get('abstract', '无')}")
            st.write(f"📅 发布时间: {hit.get('publish_time', '无')}")
            st.write(f"🔗 来源: {hit.get('source', '无')}")
            #st.write(f"📄 摘要: {hit.get('content', '无')}")
            content = hit.get('content', '') or hit.get('abstract', '')
            if content:
                with st.spinner("正在生成摘要..."):
                    summary = get_summary_qianwen(content)
            else:
                summary = "无内容"
            st.write(f"📝 千问摘要: {summary}")
            # 关键词数组处理
            keywords = hit.get('keyword', [])
            if isinstance(keywords, list):
                keywords = ', '.join(keywords)
            st.write(f"🔑 关键词: {keywords if keywords else '无'}")
            # PDF 链接
            pdf_link = hit.get('pdf_link')
            if pdf_link:
                st.markdown(f"[📎 PDF链接]({pdf_link})")
            # 文件直链
            file_url = hit.get('file_url')
            if file_url:
                st.markdown(f"[📁 文件下载]({file_url})")
            st.divider()
    elif not results:
        st.info("未找到匹配结果，请尝试其他关键词")



#streamlit run test.py