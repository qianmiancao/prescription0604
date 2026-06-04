# --- 1. 核心补丁：必须在最顶端，解决 Linux 环境 SQLite 版本问题 ---
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

# 数据库存储路径 (Streamlit Cloud 建议使用相对路径)
DB_PATH = "./chroma_db_storage"

# --- 3. 核心逻辑类 ---

class KnowledgeManager:
    def __init__(self, model_name, db_path):
        self.db_path = db_path
        # 预先加载嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = None
        self.init_db()

    def init_db(self):
        """初始化数据库，增加异常处理"""
        try:
            self.vectorstore = Chroma(
                collection_name="pharmacy_v1",
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )
        except Exception as e:
            # 如果数据库文件损坏导致初始化失败，尝试删除重建
            if os.path.exists(self.db_path):
                shutil.rmtree(self.db_path)
            os.makedirs(self.db_path, exist_ok=True)
            self.vectorstore = Chroma(
                collection_name="pharmacy_v1",
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )

    def upload_docs(self, file_path, file_name):
        """解析 PDF 并存入"""
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            splits = splitter.split_documents(docs)
            
            # 注入元数据
            for s in splits:
                s.metadata["source"] = file_name
            
            self.vectorstore.add_documents(splits)
            return len(splits)
        except Exception as e:
            st.error(f"解析文件 {file_name} 失败: {e}")
            return 0

    def retrieve_context(self, query):
        if not query or not self.vectorstore: return ""
        try:
            results = self.vectorstore.similarity_search(query, k=5)
            return "\n\n".join([f"【来源：{res.metadata.get('source', '未知')}】\n{res.page_content}" for res in results])
        except Exception as e:
            return f"检索出错: {e}"

    def get_stats(self):
        """安全地获取已存文件列表"""
        try:
            data = self.vectorstore.get()
            if not data or 'metadatas' not in data or not data['metadatas']:
                return []
            sources = {m['source'] for m in data['metadatas'] if m and 'source' in m}
            return sorted(list(sources))
        except:
            return []

    def clear_db(self):
        """彻底清空数据库目录并重启"""
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
        
        输出要求：
        1. [综合评估得分]：0-100（数字）。
        2. [风险等级]：低/中/高。
        3. [分析详情]：分点说明。
        4. [修改建议]：具体的调整方案。"""
        
        prompt = ChatPromptTemplate.from_template(
            system_prompt + "\n\n【参考资料】:\n{context}\n\n【处方数据】:\n{prescription}"
        )
        chain = prompt | self.llm
        return chain.invoke({
            "context": context if context else "未匹配到相关说明书资料。",
            "prescription": json.dumps(prescription_json, ensure_ascii=False, indent=2)
        }).content

# --- 4. 辅助工具 ---

def extract_score(text):
    match = re.search(r'综合评估得分[\]：:]*\s*(\d+)', text)
    return int(match.group(1)) if match else None

@st.cache_resource
def load_km():
    return KnowledgeManager("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", DB_PATH)

# --- 5. Streamlit UI ---

def main():
    st.set_page_config(page_title="AI 药师审方系统", layout="wide", page_icon="💊")
    
    # 初始化
    km = load_km()
    if 'audit_report' not in st.session_state: st.session_state['audit_report'] = ""
    if 'audit_score' not in st.session_state: st.session_state['audit_score'] = None

    # --- 侧边栏 ---
    with st.sidebar:
        st.title("🔐 系统管理")
        api_key = st.text_input("DeepSeek API Key:", type="password")
        
        st.divider()
        st.header("📂 药品库")
        files = st.file_uploader("上传 PDF", type="pdf", accept_multiple_files=True)
        
        c1, c2 = st.columns(2)
        if files and c1.button("✨ 同步知识"):
            with st.spinner("解析中..."):
                for f in files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.getvalue())
                        km.upload_docs(tmp.name, f.name)
                    os.unlink(tmp.name)
                st.success("同步成功")
                st.rerun()
        
        if c2.button("🗑️ 清空库"):
            km.clear_db()
            st.rerun()

        sources = km.get_stats()
        if sources:
            st.info(f"已加载 {len(sources)} 份说明书")
            for s in sources: st.caption(f"· {s}")

    # --- 主界面 ---
    st.title("🏥 药剂科 AI 处方审核平台")
    
    if not api_key:
        st.info("请先在左侧侧边栏配置 API Key 以启用 AI。")
        st.stop()

    agent = PharmacyAgent(api_key)
    
    col_l, col_r = st.columns([1, 1.3])

    with col_l:
        st.subheader("📋 录入处方")
        with st.form("p_form"):
            r1 = st.columns(2)
            age = r1[0].number_input("年龄", value=25)
            weight = r1[1].number_input("体重 (kg)", value=60.0) if age < 18 else None
            
            diag = st.text_input("临床诊断", value="急性支气管炎")
            insu = st.selectbox("医保类型", ["统筹医保", "自费", "门诊大病"])
            
            st.divider()
            med = st.text_input("药品名称", value="二甲双胍")
            dose = st.text_input("单次剂量", value="0.5g")
            freq = st.text_input("给药频次", value="一日两次")
            
            if st.form_submit_button("🧪 提交审核"):
                with st.spinner("AI 正在分析..."):
                    context = km.retrieve_context(med)
                    p_data = {
                        "patient": {"age": age, "weight": weight, "diagnosis": diag, "insurance": insu},
                        "medication": {"name": med, "dosage": dose, "frequency": freq}
                    }
                    report = agent.audit(p_data, context)
                    st.session_state['audit_report'] = report
                    st.session_state['audit_score'] = extract_score(report)
                    st.rerun()

    with col_r:
        st.subheader("📝 审核报告")
        if st.session_state['audit_report']:
            score = st.session_state['audit_score']
            if score is not None:
                st.metric("安全评分", f"{score}/100")
            
            # 可编辑区域
            edited = st.text_area("编辑报告内容", value=st.session_state['audit_report'], height=450)
            st.session_state['audit_report'] = edited
            
            st.download_button("📥 导出报告", edited, file_name=f"审核报告_{med}.txt")
        else:
            st.info("尚未生成结果。")

if __name__ == "__main__":
    main()
