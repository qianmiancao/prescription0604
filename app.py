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

DB_PATH = "./drug_db"
os.makedirs(DB_PATH, exist_ok=True)

class KnowledgeManager:
    def __init__(self, model_name, db_path):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )

    def upload_docs(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        splits = splitter.split_documents(docs)
        self.vectorstore.add_documents(splits)
        return len(splits)

    def retrieve_context(self, query):
        # 增加搜索深度，提高匹配率
        results = self.vectorstore.similarity_search(query, k=4)
        return "\n".join([res.page_content for res in results])

class PharmacyAgent:
    def __init__(self, api_key):
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key.strip(),
            base_url="https://api.deepseek.com",
            temperature=0
        )

    def audit(self, prescription_json, context):
        system_prompt = """你是一位资深临床药师。请根据【参考资料】审核【处方数据】。
        
        审核逻辑：
        1. 如果年龄>=18岁，视为成人，重点关注诊断匹配和常规剂量。
        2. 如果年龄<18岁，视为儿童，必须严格根据体重核算剂量。
        3. 如果【参考资料】为空，请根据你的临床知识储备进行初步审核，但务必提醒用户“未匹配到说明书，以下仅供参考”。
        4. 检查适应症、禁忌症及医保报销合规性。"""
        
        prompt = ChatPromptTemplate.from_template(
            system_prompt + "\n\n【参考资料】:\n{context}\n\n【处方数据】:\n{prescription}"
        )
        chain = prompt | self.llm
        return chain.invoke({
            "context": context if context else "未找到相关说明书内容",
            "prescription": json.dumps(prescription_json, ensure_ascii=False, indent=2)
        }).content

@st.cache_resource
def get_km():
    return KnowledgeManager("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", DB_PATH)

# --- 3. Streamlit UI ---

def main():
    st.set_page_config(page_title="AI 药师审方系统", layout="wide")
    km = get_km()

    # 登录逻辑
    st.sidebar.title("🔐 系统登录")
    if 'api_key' not in st.session_state: st.session_state['api_key'] = ""
    input_key = st.sidebar.text_input("DeepSeek API Key:", type="password", value=st.session_state['api_key'])
    
    if not input_key:
        st.info("请在左侧侧边栏输入 API Key 以开始。")
        st.stop()
    
    st.session_state['api_key'] = input_key
    agent = PharmacyAgent(input_key)

    st.title("🏥 药剂科 AI 处方审核平台")

    # 侧边栏知识库
    with st.sidebar:
        st.header("📂 知识库管理")
        files = st.file_uploader("上传 PDF 说明书", type="pdf", accept_multiple_files=True)
        if files and st.button("✨ 立即同步知识"):
            with st.spinner("正在学习说明书..."):
                for f in files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t:
                        t.write(f.getvalue())
                        km.upload_docs(t.name)
                    os.unlink(t.name)
            st.success("同步完成！")

    # 主界面布局
    col_in, col_out = st.columns([1, 1.2])

    with col_in:
        st.subheader("📋 录入处方")
        with st.form("my_form"):
            r1 = st.columns(2)
            age = r1[0].number_input("年龄", value=25, min_value=0)
            
            # --- 动态显示体重输入 ---
            if age < 18:
                weight = r1[1].number_input("体重 (kg)", value=18.0)
                is_child = True
            else:
                r1[1].info("成人无需输入体重")
                weight = None
                is_child = False
            
            r2 = st.columns(2)
            diag = r2[0].text_input("临床诊断", value="急性支气管炎")
            insu = r2[1].selectbox("医保类型", ["统筹医保", "自费", "门诊大病"])
            
            st.divider()
            med = st.text_input("药品名称", value="头孢克肟")
            dose = st.text_input("单次剂量", value="100mg")
            freq = st.text_input("给药频次", value="一日两次")
            
            submit = st.form_submit_button("🧪 提交审核")

    with col_out:
        st.subheader("📝 审核结果")
        if submit:
            # 1. 构造处方数据
            prescription = {
                "patient": {"age": age, "weight": weight, "diagnosis": diag, "insurance": insu, "type": "儿童" if is_child else "成人"},
                "medication": {"name": med, "dosage": dose, "frequency": freq}
            }
            
            # 2. 检索知识
            with st.spinner("正在检索说明书..."):
                context = km.retrieve_context(med)
                
                # 调试辅助：在界面显示检索到了哪些内容
                with st.expander("🔍 知识库匹配结果预览"):
                    if context:
                        st.write(context)
                    else:
                        st.warning("警告：向量数据库中未找到关于该药品的具体条目。请确保已同步说明书。")

            # 3. AI 审核
            with st.spinner("AI 药师正在分析..."):
                report = agent.audit(prescription, context)
                st.markdown(report)
        else:
            st.info("待录入...")

if __name__ == "__main__":
    main()
