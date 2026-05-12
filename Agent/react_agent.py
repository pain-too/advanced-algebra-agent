# 第三方库
from langchain.agents import create_agent
# 项目模块
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from tools.agent_tools import ds_knowledge_search, ds_concept_compare, ds_chapter_summary


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[ds_knowledge_search, ds_concept_compare, ds_chapter_summary],
        )

    def execute_stream(self, query):
        input_dict = {
            "messages": [
                {"role": "user", "content": query}
            ]
        }
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk['messages'][-1]
            yield latest_message.content.strip() + '\n'