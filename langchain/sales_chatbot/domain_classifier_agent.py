from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class DomainClassifierAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.5)

    def classify(self, query, history):
        messages = [SystemMessage(content="你是一個頂尖顧問公司的金牌業務，能準確地判斷客戶的問題屬於下列哪一個領域：house, router, tv 或者 unknown。輸出格式請參考 domain is router")]

        # 将对话历史加入到消息列表中
        if history:
            for conversation in history:
                usr_msg, sys_reply = conversation
                messages.append(HumanMessage(content=usr_msg))
                messages.append(SystemMessage(content=sys_reply))
        
        # 添加当前查询
        messages.append(HumanMessage(content=query))
        
        response = self.llm.invoke(messages).content
        
        # 基于模型的回答来判断领域
        if "router" in response.lower():
            return "router"
        elif "house" in response.lower():
            return "real_estate"
        elif "tv" in response.lower():
            return "tv"
        else:
            return "unknown"
