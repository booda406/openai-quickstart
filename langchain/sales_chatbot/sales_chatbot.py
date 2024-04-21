import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from sales_bot_manager import SalesBotManager
from domain_classifier_agent import DomainClassifierAgent

def create_vector_databases(text_files: list, database_dirs: list):
    """
    为给定的文本文件创建向量数据库。
    
    :param text_files: 要处理的文本文件路径列表。
    :param database_dirs: 每个文本文件相应的数据库存储目录列表。
    """
    # 确保OpenAI的Embeddings模型初始化一次
    embeddings_model = OpenAIEmbeddings()

    for text_file, db_dir in zip(text_files, database_dirs):
        # 读取文本文件内容
        with open(text_file, 'r', encoding='utf-8') as file:
            texts = file.read()

        text_splitter = CharacterTextSplitter(        
            separator = r'\d+\.',
            chunk_size = 100,
            chunk_overlap  = 0,
            length_function = len,
            is_separator_regex = True,
        )

        docs = text_splitter.create_documents([texts])

        db = FAISS.from_documents(docs, OpenAIEmbeddings())

        db.save_local(db_dir)

def sales_chat(message, history, session_id):
    print(f"[message]{message}")
    print(f"[history]{history}")

    enable_chat = True
    
    classifier_agent = DomainClassifierAgent()
    domain = classifier_agent.classify(message, history)
    print(f" domain is {domain}")

    if domain == "router":
        selected_bot = router_bot
    elif domain == "real_estate":
        selected_bot = real_estate_bot
    elif domain == "tv":
        selected_bot = tv_bot
    else:
        return "对不起，我不确定您的问题属于哪个领域，请尝试提供更多信息。"

    ans = selected_bot({"query": message})
    
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="销售金牌",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    text_files = ['real_estate_sales_data.txt', 'router_data.txt', 'tv_data.txt']
    database_dirs = ['real_estate_sales_vector_db', 'router_vector_db', 'tv_vector_db']
    create_vector_databases(text_files, database_dirs)

    bot_manager = SalesBotManager()
    
    # 初始化不同领域的聊天机器人
    bot_manager.initialize_bot("real_estate", "real_estate_sales_vector_db")
    bot_manager.initialize_bot("router", "router_vector_db")
    bot_manager.initialize_bot("tv", "tv_vector_db")
    
    # 根据需要获取并使用特定领域的聊天机器人
    real_estate_bot = bot_manager.get_bot("real_estate")
    router_bot = bot_manager.get_bot("router")
    tv_bot = bot_manager.get_bot("tv")

    # 启动 Gradio 服务
    launch_gradio()
