import re
import streamlit as st
from meilisearch import Client
import time
import requests
import json
import os
from openai import OpenAI

# 加载配置文件
def load_config():
    """加载配置文件"""
    config_path = "config.json"
    if not os.path.exists(config_path):
        st.error(f"配置文件 {config_path} 不存在！")
        st.stop()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"读取配置文件失败：{str(e)}")
        st.stop()

# 加载配置
config = load_config()

# 初始化 OpenAI 客户端（通义千问）
client = OpenAI(
    api_key=config["openai"]["api_key"],
    base_url=config["openai"]["base_url"],
)

# 初始化 Meilisearch 客户端
meili_client = Client(
    config["meilisearch"]["url"], 
    config["meilisearch"]["api_key"]
)



def get_embedding(query):
    """获取文本的向量嵌入表示"""
    url = config["embedding"]["url"]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['embedding']['api_key']}"
    }
    payload = {
        "texts": [query],
        "model": config["embedding"]["model"]
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    # 取第一个 embedding
    return data["data"][0]["embedding"]



def search_meilisearch_hybrid(query, knowledge_base, top_k, semantic_ratio):
    """使用混合搜索（关键词+语义）在 Meilisearch 中搜索文档"""
    try:
        # 获取指定知识库的索引
        index = meili_client.index(knowledge_base)
        # 获取查询文本的向量嵌入
        embedding = get_embedding(query)
        # 执行混合搜索
        results = index.search(
            query,
            {
                "vector": embedding,
                "hybrid": {
                    "semanticRatio": 1 - semantic_ratio,  # 语义搜索权重
                    "embedder": "bge_m3"  # 嵌入模型名称
                },
                "limit": top_k  # 返回结果数量限制
            }
        )
        return results.get("hits", []), True
    except Exception as e:
        st.error(f"连接 Meilisearch 失败：{str(e)}")
        return [], False

def get_summary_qianwen(text):
    """使用通义千问生成文本摘要"""
    prompt = f"请用中文对以下内容生成简明摘要,只需返回摘要，别的任何说明都不返回：\n{text}"
    try:
        response = client.chat.completions.create(
            model=config["openai"]["model"], 
            messages=[
                {"role": "system", "content": "你是一个专业的中文摘要助手，只需返回摘要，别的任何说明都不返回。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # 控制生成文本的随机性
            max_tokens=128    # 限制生成文本的最大长度
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"摘要生成失败: {e}"
    
def get_keywords_qianwen(text):
    """使用通义千问生成文本关键词"""
    prompt = f"请用中文对以下内容生成关键词,只需返回关键词，别的任何说明都不返回：\n{text}"
    try:
        response = client.chat.completions.create(
            model=config["openai"]["model"], 
            messages=[
                {"role": "system", "content": "你是一个专业的中文关键词助手，只会返回关键词，别的任何说明都不返回。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # 控制生成文本的随机性
            max_tokens=128    # 限制生成文本的最大长度
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"关键词生成失败: {e}"




# 侧边栏配置
with st.sidebar:
    st.header("搜索设置")
    # 知识库选择（需与 Meilisearch 中的索引名一致）
    knowledge_base = st.selectbox(
        "知识库", 
        [config["search"]["default_knowledge_base"]], 
        help="选择要搜索的知识库"
    )
    # 语义系数滑块（控制语义搜索与关键词搜索的权重比例）
    semantic_ratio = st.slider(
        "SemanticRatio", 
        min_value=0.0, 
        max_value=1.0, 
        value=config["search"]["default_semantic_ratio"], 
        step=0.1,
        help="调整语义匹配权重，0为纯关键词搜索，1为纯语义搜索"
    )
    # 返回结果数量
    top_k = st.number_input(
        "返回结果数量(topK)", 
        min_value=1, 
        max_value=config["search"]["max_top_k"], 
        value=config["search"]["default_top_k"], 
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
search_query = st.text_input("请输入搜索关键词", value="AI", help="支持关键词、短语搜索，系统会自动进行语义理解")
search_btn = st.button("搜索", type="primary")

# 搜索逻辑
if search_btn:
    # 记录搜索开始时间
    start_time = time.time()
    
    # 调用 Meilisearch 混合搜索
    results, success = search_meilisearch_hybrid(search_query, knowledge_base, top_k, semantic_ratio)
    
    # 计算搜索耗时
    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000
    
    # 更新侧边栏状态显示
    search_time_placeholder.markdown(f"### 搜索耗时：{duration_ms:.2f} ms")
    result_count_placeholder.markdown(f"### 返回结果数：{len(results)} 条")
    
    # 展示搜索结果
    if success and results:
        for i, hit in enumerate(results, start=1):
            # 显示文档标题和基本信息
            st.markdown(f"### {i}. {hit.get('title', '无标题')}")
            st.write(f"🆔 SHA256: {hit.get('_sha256', hit.get('file_sha256', '无'))}")
            st.write(f"👤 作者: {hit.get('author', '无')}")
            st.write(f"🏢 机构: {hit.get('organization', '无')}")
            st.write(f"📊 行业: {hit.get('industry', '无')}")
            st.write(f"📅 发布时间: {hit.get('publish_time', '无')}")
            st.write(f"🔗 来源: {hit.get('source', '无')}")
            
            # 获取文档内容并生成AI摘要和关键词
            content = hit.get('content', '') or hit.get('abstract', '')
            if content:
                with st.spinner("正在生成摘要和关键词..."):
                    summary = get_summary_qianwen(content)
                    keywords = get_keywords_qianwen(content)
            else:
                summary = '无内容'
                keywords = '无关键词'
            
            # 显示AI生成的摘要和关键词（markdown格式需要两个以上空格+\n才能换行）
            st.write(f"📝 千问摘要:  \n{summary}")
            st.write(f"🔑 千问关键词:  \n{keywords}")
            
            # 显示文档链接
            pdf_link = hit.get('pdf_link')
            if pdf_link:
                st.markdown(f"[📎 PDF链接]({pdf_link})")
            
            file_url = hit.get('file_url')
            if file_url:
                st.markdown(f"[📁 文件下载]({file_url})")
            
            st.divider()  # 分隔线
    elif not results:
        st.info("未找到匹配结果，请尝试其他关键词")



# 运行命令: streamlit run ai.py
