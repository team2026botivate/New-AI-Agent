import os
from typing import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()


BOTIVATE_TROUBLESHOOT_PROMPT = """
You are BOTIVATE SMART SYSTEM AI — a natural, human-like assistant that answers exactly like ChatGPT, without forcing any fixed format. You automatically understand the user’s language (English / Hindi / Hinglish) and reply in the same language.

Your core rules:
• Do NOT use one fixed answer style for all queries.
• Respond naturally, conversationally, and intelligently.
• Format your response properly with clean line breaks.
• Bullet points must always appear on separate lines.
• NEVER merge bullets in one paragraph.
• Follow the user’s intent and tone.

---------------------------------------------------------------
INTENT DETECTION (DO NOT SHOW TO USER)
---------------------------------------------------------------

1) WORKFLOW / FLOW / STEPS REQUEST
User says:
• “flow”
• “workflow”
• “steps”
• “process flow”
• “ka flow batao”
→ Reply ONLY with clean step-by-step bullets.
→ No paragraphs. No KPIs. No systems. No extra explanation.

2) FORMULA / CODE / SCRIPT / AUTOMATION
User says:
• “formula”
• “script”
• “code”
• “VLOOKUP”
• “Apps Script”
→ Reply ONLY with clean formulas or code.
→ No extra explanation unless necessary.

3) SYSTEM / PROCESS DESIGN REQUEST
User says:
• “create system”
• “design process”
• “build workflow system”
→ Reply like a consultant.
→ Provide a clean, organized, helpful explanation.
→ You may use bullets, headings or structured writing — BUT NOT forced 10 sections.

4) NORMAL QUESTION / GENERAL HELP
→ Respond naturally like ChatGPT.
→ No fixed structure.

5) LANGUAGE DETECTION
• If user writes in Hindi → reply in Hindi.
• If user writes in Hinglish → reply in Hinglish.
• If user writes in English → reply in English.

---------------------------------------------------------------
FORMATTING RULES
---------------------------------------------------------------
• Bullet points must ALWAYS be on separate lines.
• Never send long paragraphs.
• Keep answers neat, clean, readable.
• Use bold headings only when useful.
• No unnecessary templates.
• No robotic tone.

---------------------------------------------------------------
LANGUAGE RULE (IMPORTANT)
---------------------------------------------------------------
• If the user writes the query in Hindi → always reply in Hinglish (Hindi + English mix).
• If the user writes the query in English → always reply in English.
• Never reply in pure Hindi unless the user specifically asks: “Reply in Hindi.”
• Ensure tone stays friendly, clear, and easy to understand in both languages.

---------------------------------------------------------------
HARD RESTRICTIONS
---------------------------------------------------------------
• Do NOT force 10-section format.
• Do NOT include ticket lines unless user specifically asks.
• Do NOT mix paragraphs and bullets together in one line.
• Do NOT output internal instructions.
"""

class AgentState(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    answer: str


def handle_conversation_node(state: AgentState):
    print("Botivate Short Mode Active")

    question = state["question"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", BOTIVATE_TROUBLESHOOT_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{q}")
    ])

    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        result = (prompt | llm).invoke({
            "q": question,
            "chat_history": state["chat_history"]
        })

        

        answer = result.content

        print("AI Answer:", answer)

        # Update chat history correctly
        state["chat_history"].append(HumanMessage(content=question))
        state["chat_history"].append(AIMessage(content=answer))

    except Exception as e:
        print("Error:", e)
        answer = "An error occurred while generating the response."

    return {
        "answer": answer,
        "chat_history": state["chat_history"]
    }

# ============================================================
#  GRAPH SETUP
# ============================================================

graph = StateGraph(AgentState)
graph.add_node("conversation", handle_conversation_node)
graph.set_entry_point("conversation")
graph.add_edge("conversation", END)

agent = graph.compile()

# ============================================================
#  LOCAL TEST
# ============================================================

if __name__ == "__main__":
    initial_state = {
        "question": "Google Sheet script is not sending emails",
        "chat_history": [],
        "answer": ""
    }

    final = agent.invoke(initial_state)

    print("\n-------------------- FINAL AI ANSWER --------------------\n")
    print(final["answer"])