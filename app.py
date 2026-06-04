# --- 1. 核心兼容性补丁 (针对 Linux/Streamlit Cloud 环境) ---
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
DB_PATH = "./drug_db"

# --- 3. 核心逻辑类 ---

class KnowledgeManager:
    def __init__(self, model_name, db_path):
        self.db_path = db_path
        # 初始化嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
        self.init_db()

    def init_db(self):
        """初始化或重载数据库连接"""
        self.vectorstore = Chroma(
            collection_name="pharmacy_docs",
            persist_directory=self.db_path,
            embedding_function=self.embeddings
        )

    def upload_docs(self, file_path, file_name):
        """解析PDF并存入数据库，同时注入文件名元数据"""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        # 优化切分大小，增加重叠度以防核心信息被切断
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = splitter.split_documents(docs)
        
        # 注入元数据，方便检索时追溯来源
        for s in splits:
            s.metadata["source"] = file_name
            
        self.vectorstore.add_documents(splits)
        return len(splits)

    def retrieve_context(self, query):
        """检索最相关的5条知识片段"""
        if not query: return ""
        results = self.vectorstore.similarity_search(query, k=5)
        # 格式化输出：带上来源文件名
        return "\n\n".join([f"【来源：{res.metadata.get('source', '未知')}】\n{res.page_content}" for res in results])

    def get_stats(self):
        """获取当前库中已存储的所有文件名清单"""
        try:
            data = self.vectorstore.get()
            if not data or not data['metadatas']:
                return []
            sources = set(m['source'] for m in data['metadatas'] if 'source' in m)
            return sorted(list(sources))
        except:
            return []

    def clear_db(self):
        """彻底清空本地数据库目录"""
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
        system_prompt = """你是一位资深临床药师。请根据【参考资料】审核【处方数据】。
        
        审核逻辑：
        1. 年龄>=18岁视为成人，关注诊断匹配度；年龄<18岁必须核算体重剂量。
        2. 若参考资料为空，请基于临床常识初步审核并提醒用户“未找到说明书”。
        3. 检查适应症、用法用量、禁忌症及医保合规性。"""
        
        prompt = ChatPromptTemplate.from_template(
            system_prompt + "\n\n【参考资料】:\n{context}\n\n【处方数据】:\n{prescription}"
        )
        chain = prompt | self.llm
        return chain.invoke({
            "context": context if context else "未在数据库中找到该药品的参考资料。",
            "prescription": json.dumps(prescription_json, ensure_ascii=False, indent=2)
        }).content

# --- 4. 资源缓存 ---

@st.cache_resource
def get_km():
    return KnowledgeManager("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", DB_PATH)

# --- 5. Streamlit UI 界面 ---

def main():
    st.set_page_config(page_title="AI 药师审方系统", layout="wide", page_icon="💊")
    km = get_km()

    # --- 侧边栏：登录与知识库管理 ---
    with st.sidebar:
        st.title("🔐 系统设置")
        
        # API Key 管理
        if 'api_key' not in st.session_state: st.session_state['api_key'] = ""
        key = st.text_input("DeepSeek API Key:", type="password", value=st.session_state['api_key'])
        if key: st.session_state['api_key'] = key
        
        st.divider()
        
        # 知识库管理
        st.header("📂 知识库管理")
        files = st.file_uploader("上传药品说明书 (PDF)", type="pdf", accept_multiple_files=True)
        
        c1, c2 = st.columns(2)
        if files and c1.button("✨ 同步知识"):
            with st.spinner("正在解析并存入库..."):
                for f in files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t:
                        t.write(f.getvalue())
                        km.upload_docs(t.name, f.name)
                    os.unlink(t.name)
                st.success("同步成功！")
                st.rerun() # 强制刷新以更新清单

        if c2.button("🗑️ 清空所有"):
            km.clear_db()
            st.warning("所有知识已清除")
            st.rerun()

        # 展示已存在的文件清单
        sources = km.get_stats()
        if sources:
            st.info(f"已学习 {len(sources)} 份说明书：")
            for s in sources:
                st.caption(f"· {s}")
        else:
            st.caption("当前知识库为空")

    # --- 主界面逻辑 ---
    if not st.session_state['api_key']:
        st.info("👋 请在左侧侧边栏输入您的 DeepSeek API Key 以激活系统。")
        st.stop()
    
    agent = PharmacyAgent(st.session_state['api_key'])

    st.title("🏥 药剂科 AI 处方审核平台")
    st.markdown("---")

    col_in, col_out = st.columns([1, 1.2])

    # 左侧：处方录入
    with col_in:
        st.subheader("📋 录入处方")
        with st.form("audit_form"):
            r1 = st.columns(2)
            age = r1[0].number_input("年龄", value=25, min_value=0)
            
            # 成人/儿童动态逻辑
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
            med = st.text_input("药品名称", value="来那度胺")
            dose = st.text_input("单次剂量", value="25mg")
            freq = st.text_input("给药频次", value="一日一次")
            
            submit = st.form_submit_button("🧪 提交审核")

    # 右侧：审核结果
    with col_out:
        st.subheader("📝 审核结果")
        if submit:
            # 构建处方 JSON
            prescription = {
                "patient": {
                    "age": age, 
                    "weight": weight, 
                    "diagnosis": diag, 
                    "insurance": insu, 
                    "type": "儿童" if is_child else "成人"
                },
                "medication": {
                    "name": med, 
                    "dosage": dose, 
                    "frequency": freq
                }
            }
            
            # 1. 检索阶段
            with st.spinner("🔍 正在检索相关说明书内容..."):
                context = km.retrieve_context(med)
                
                with st.expander("📄 查看检索到的原始片段"):
                    if context:
                        st.markdown(context)
                    else:
                        st.warning(f"未能匹配到关于 '{med}' 的具体内容。")

            # 2. AI 审核阶段
            with st.spinner("🤖 AI 药师正在分析并生成报告..."):
                try:
                    report = agent.audit(prescription, context)
                    st.markdown("### 诊断分析报告")
                    st.info(report)
                    st.download_button(
                        "📥 导出审核报告", 
                        report, 
                        file_name=f"审核报告_{med}_{datetime.now().strftime('%Y%m%d')}.txt"
                    )
                except Exception as e:
                    st.error(f"审核过程出错：{e}")
        else:
            st.info("请在左侧填写处方信息并点击“提交审核”。")

if __name__ == "__main__":
    main()
