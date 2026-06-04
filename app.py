# --- 1. 核心兼容性补丁 ---
try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
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

# --- 2. 基础环境配置 ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

DB_PATH = "./drug_db"

# --- 3. 核心逻辑类 ---

class KnowledgeManager:
    def __init__(self, model_name, db_path):
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
        self.init_db()

    def init_db(self):
        self.vectorstore = Chroma(
            collection_name="pharmacy_docs",
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
        if not query: return ""
        try:
            results = self.vectorstore.similarity_search(query, k=5)
            return "\n\n".join([f"【来源：{res.metadata.get('source', '未知')}】\n{res.page_content}" for res in results])
        except: return ""

    def get_stats(self):
        try:
            data = self.vectorstore.get()
            if not data or 'metadatas' not in data: return []
            sources = set(m['source'] for m in data['metadatas'] if m and 'source' in m)
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
        
        审核要求：
        1. 评分标准（0-100分）：100分为完全合规安全；60分以下为存在严重风险。
        2. 结构化输出结果：
           - [综合评估得分]：仅输出数字（0-100）。
           - [风险等级]：低/中/高。
           - [分析详情]：分条目列出适应症、用法用量、特殊人群风险。
           - [修改建议]：若有问题，请给出具体调整方案。
        3. 若参考资料不足，请根据常识审核并标注“仅供参考”。"""
        
        prompt = ChatPromptTemplate.from_template(
            system_prompt + "\n\n【参考资料】:\n{context}\n\n【处方数据】:\n{prescription}"
        )
        chain = prompt | self.llm
        return chain.invoke({
            "context": context if context else "未匹配到相关说明书资料。",
            "prescription": json.dumps(prescription_json, ensure_ascii=False, indent=2)
        }).content

# --- 4. 辅助函数 ---

def extract_score(text):
    """从 AI 文本中提取数字得分"""
    match = re.search(r'综合评估得分[\]：:]*\s*(\d+)', text)
    if match:
        return int(match.group(1))
    return None

@st.cache_resource
def get_km():
    return KnowledgeManager("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", DB_PATH)

# --- 5. Streamlit UI ---

def main():
    st.set_page_config(page_title="AI 药师审方系统", layout="wide", page_icon="💊")
    km = get_km()

    # 初始化 session_state
    if 'audit_report' not in st.session_state: st.session_state['audit_report'] = ""
    if 'audit_score' not in st.session_state: st.session_state['audit_score'] = None
    if 'api_key' not in st.session_state: st.session_state['api_key'] = ""

    # --- 侧边栏 ---
    with st.sidebar:
        st.title("🔐 系统管理")
        user_key = st.text_input("DeepSeek API Key:", type="password", value=st.session_state['api_key'])
        if user_key: st.session_state['api_key'] = user_key
        
        st.divider()
        st.header("📂 药品说明书库")
        files = st.file_uploader("上传 PDF 说明书", type="pdf", accept_multiple_files=True)
        
        c1, c2 = st.columns(2)
        if files and c1.button("✨ 同步"):
            with st.spinner("处理中..."):
                for f in files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.getvalue())
                        km.upload_docs(tmp.name, f.name)
                    os.unlink(tmp.name)
                st.success("同步完成！")
                st.rerun()
        if c2.button("🗑️ 清空库"):
            km.clear_db()
            st.rerun()

        doc_list = km.get_stats()
        if doc_list:
            st.info(f"库中已有 {len(doc_list)} 份文件")
            for d in doc_list: st.caption(f"· {d}")

    # --- 主界面 ---
    if not st.session_state['api_key']:
        st.info("请先在左侧配置 API Key。")
        st.stop()
    
    agent = PharmacyAgent(st.session_state['api_key'])

    st.title("🏥 药剂科 AI 处方审核平台")
    st.markdown("---")

    col_form, col_result = st.columns([1, 1.3])

    # --- 左侧：录入处方 ---
    with col_form:
        st.subheader("📋 录入处方信息")
        with st.form("prescription_form"):
            r1 = st.columns(2)
            age = r1[0].number_input("年龄", value=25, min_value=0)
            weight = r1[1].number_input("体重 (kg)", value=60.0) if age < 18 else None
            if age >= 18: r1[1].info("成人无需输入体重")
            
            r2 = st.columns(2)
            diagnosis = r2[0].text_input("临床诊断", value="急性支气管炎")
            insurance = r2[1].selectbox("医保类型", ["统筹医保", "自费", "门诊大病"])
            
            st.divider()
            med_name = st.text_input("药品名称", value="来那度胺")
            dosage = st.text_input("单次剂量", value="25mg")
            freq = st.text_input("给药频次", value="一日一次")
            
            if st.form_submit_button("🧪 提交 AI 审核"):
                # 构造数据
                prescription = {
                    "patient": {"age": age, "weight": weight, "diagnosis": diagnosis, "insurance": insurance},
                    "medication": {"name": med_name, "dosage": dosage, "frequency": freq}
                }
                
                with st.spinner("AI 正在查阅说明书并审核..."):
                    context = km.retrieve_context(med_name)
                    raw_report = agent.audit(prescription, context)
                    
                    # 更新状态
                    st.session_state['audit_report'] = raw_report
                    st.session_state['audit_score'] = extract_score(raw_report)
                    st.rerun()

    # --- 右侧：审核报告与评分 ---
    with col_result:
        st.subheader("📝 审核报告")
        
        if st.session_state['audit_report']:
            # 1. 显示打分
            score = st.session_state['audit_score']
            if score is not None:
                color = "normal" if score >= 80 else "inverse" # 颜色反馈
                st.metric(label="处方安全评估得分", value=f"{score} / 100", delta=f"{'安全' if score>=80 else '高风险'}")
                if score < 60:
                    st.error("⚠️ 该处方存在重大安全隐患，建议药师人工介入！")
                elif score < 80:
                    st.warning("💡 该处方存在部分瑕疵或风险，请仔细复核。")
                else:
                    st.success("✅ 该处方初步评估为安全。")

            # 2. 显示可编辑的报告
            st.markdown("#### AI 生成意见 (点击下方框内可手动修改):")
            # 用户可以在这里直接修改报告
            edited_report = st.text_area(
                label="审核详情编辑器",
                value=st.session_state['audit_report'],
                height=450,
                label_visibility="collapsed"
            )
            
            # 同步修改到 session_state
            st.session_state['audit_report'] = edited_report

            # 3. 导出功能
            st.divider()
            c1, c2 = st.columns(2)
            c1.download_button(
                label="📥 导出最终报告 (含人工修改)",
                data=st.session_state['audit_report'],
                file_name=f"审核报告_{datetime.now().strftime('%Y%m%d%H%M')}.txt",
                mime="text/plain"
            )
            if c2.button("🔄 重新生成结果"):
                st.session_state['audit_report'] = ""
                st.rerun()
        else:
            st.info("尚未提交审核，请在左侧填写处方信息。")

if __name__ == "__main__":
    main()
