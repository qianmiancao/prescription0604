# --- 1. 核心补丁：解决 Linux 环境 SQLite 版本问题 ---
import sys
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import os
import json
import warnings
import tempfile
import shutil
import re
from datetime import datetime

# --- 2. 基础配置 ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# 数据库存储路径
DB_PATH = "./chroma_db_storage"

# --- 3. 逻辑类定义 ---

class KnowledgeManager:
    def __init__(self, model_name, db_path):
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = None
        self.init_db()

    def init_db(self):
        try:
            self.vectorstore = Chroma(
                collection_name="pharmacy_v1",
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )
        except Exception:
            if os.path.exists(self.db_path):
                shutil.rmtree(self.db_path)
            os.makedirs(self.db_path, exist_ok=True)
            self.vectorstore = Chroma(
                collection_name="pharmacy_v1",
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )

    def upload_docs(self, file_path, file_name):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = splitter.split_documents(docs)
        for s in splits:
            s.metadata["source"] = file_name
        self.vectorstore.add_documents(splits)
        return len(splits)

    def retrieve_context(self, query):
        if not query or not self.vectorstore: return ""
        try:
            results = self.vectorstore.similarity_search(query, k=5)
            return "\n\n".join([f"【来源：{res.metadata.get('source', '未知')}】\n{res.page_content}" for res in results])
        except: return ""

    def get_stats(self):
        try:
            data = self.vectorstore.get()
            if not data or 'metadatas' not in data: return []
            sources = {m['source'] for m in data['metadatas'] if m and 'source' in m}
            return sorted(list(sources))
        except: return []

    def clear_db(self):
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
        os.makedirs(self.db_path, exist_ok=True)
        self.init_db()

class PharmacyAgent:
    def __init__(self, api_key):
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key.strip(),
            base_url="https://api.deepseek.com",
            temperature=0
        )

    def audit(self, prescription_json, context):
        system_prompt = """你是一位资深临床药师。请根据【参考资料】审核【处方数据】并给出打分。
        输出：1.[综合评估得分](0-100) 2.[风险等级] 3.[分析详情] 4.[修改建议]"""
        prompt = ChatPromptTemplate.from_template(system_prompt + "\n\n参考资料: {context}\n处方: {prescription}")
        chain = prompt | self.llm
        return chain.invoke({"context": context or "无相关资料", "prescription": json.dumps(prescription_json, ensure_ascii=False)}).content

# --- 4. 辅助函数 ---
def extract_score(text):
    match = re.search(r'综合评估得分[\]：:]*\s*(\d+)', text)
    return int(match.group(1)) if match else None

@st.cache_resource
def load_km():
    return KnowledgeManager("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", DB_PATH)

# --- 5. UI 界面 ---

def main():
    st.set_page_config(page_title="AI 药师审方系统", layout="wide", page_icon="💊")
    km = load_km()

    # 初始化 Session State
    if 'audit_report' not in st.session_state: st.session_state['audit_report'] = ""
    if 'audit_score' not in st.session_state: st.session_state['audit_score'] = None

    # --- 侧边栏 ---
    with st.sidebar:
        st.title("🔐 系统设置")
        api_key = st.text_input("DeepSeek API Key:", type="password")
        st.divider()
        st.header("📂 药品说明书库")
        files = st.file_uploader("上传 PDF", type="pdf", accept_multiple_files=True)
        c1, c2 = st.columns(2)
        if files and c1.button("✨ 同步"):
            for f in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getvalue())
                    km.upload_docs(tmp.name, f.name)
                os.unlink(tmp.name)
            st.rerun()
        if c2.button("🗑️ 清空库"):
            km.clear_db()
            st.rerun()
        sources = km.get_stats()
        if sources:
            st.info(f"已学习 {len(sources)} 份说明书")

    # --- 主界面 ---
    st.title("🏥 药剂科 AI 处方审核平台")
    if not api_key:
        st.info("请在左侧侧边栏配置 API Key 以启用 AI。")
        st.stop()

    agent = PharmacyAgent(api_key)
    col_l, col_r = st.columns([1, 1.3])

    with col_l:
        st.subheader("📋 录入处方")
        
        # --- 关键修改：将年龄和体重放在 form 外面，实现实时响应 ---
        st.markdown("##### 患者基本信息")
        r_base = st.columns(2)
        age = r_base[0].number_input("年龄", value=25, min_value=0, max_value=150)
        
        # 实时判断逻辑
        weight = None
        if age < 18:
            weight = r_base[1].number_input("体重 (kg)", value=20.0, step=0.1)
            st.caption("👦 已切换至儿童处方模式：需核算体重剂量")
        else:
            r_base[1].info("成人模式")
            st.caption("👨 已切换至成人处方模式")

        # --- 其他信息保留在 form 中 ---
        with st.form("prescription_detail_form"):
            st.markdown("##### 临床及用药详情")
            diag = st.text_input("临床诊断", value="急性支气管炎")
            insu = st.selectbox("医保类型", ["统筹医保", "自费", "门诊大病"])
            
            st.divider()
            med = st.text_input("药品名称", value="二甲双胍")
            dose = st.text_input("单次剂量", value="0.5g")
            freq = st.text_input("给药频次", value="一日两次")
            
            submitted = st.form_submit_button("🧪 提交 AI 审核")

            if submitted:
                with st.spinner("AI 药师正在分析..."):
                    context = km.retrieve_context(med)
                    p_data = {
                        "patient": {"age": age, "weight": weight, "diagnosis": diag, "insurance": insu},
                        "medication": {"name": med, "dosage": dose, "frequency": freq}
                    }
                    report = agent.audit(p_data, context)
                    st.session_state['audit_report'] = report
                    st.session_state['audit_score'] = extract_score(report)
                    # 提交后页面会刷新，数据已在 session_state 中

    with col_r:
        st.subheader("📝 审核报告")
        if st.session_state['audit_report']:
            # 评分显示
            score = st.session_state['audit_score']
            if score is not None:
                st.metric("安全评分", f"{score}/100", delta="- 风险" if score < 60 else "+ 安全")
            
            # 编辑区：用户可以直接修改结果
            edited_report = st.text_area(
                "AI 建议 (可手动编辑修改)", 
                value=st.session_state['audit_report'], 
                height=450
            )
            st.session_state['audit_report'] = edited_report # 保持修改后的内容
            
            st.divider()
            c1, c2 = st.columns(2)
            c1.download_button("📥 导出最终报告", edited_report, file_name=f"审核报告_{med}.txt")
            if c2.button("🔄 重新开始"):
                st.session_state['audit_report'] = ""
                st.rerun()
        else:
            st.info("左侧录入数据后，点击“提交 AI 审核”即可生成结果。")

if __name__ == "__main__":
    main()
