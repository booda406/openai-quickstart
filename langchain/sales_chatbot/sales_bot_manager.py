# sales_bot_manager.py

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

class SalesBotManager:
    def __init__(self):
        self.bots = {}

    def initialize_bot(self, domain: str, vector_store_dir: str):
        db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        bot = RetrievalQA.from_chain_type(llm,
                                          retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                    search_kwargs={"score_threshold": 0.8}))
        bot.return_source_documents = True

        self.bots[domain] = bot

    def get_bot(self, domain: str):
        return self.bots.get(domain)
