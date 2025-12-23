import os, json
from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from datetime import datetime
import re
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid
import threading

# ✅ OPENAI CLIENT
openai_client = OpenAI(
    timeout=60.0,
    max_retries=2
)


def extract_limit(question: str, default=5):
    match = re.search(r"\b(\d+)\b", question)
    return int(match.group(1)) if match else default


# ---------------- DATABASE ----------------
SUPABASE_DB_URI = os.getenv("DATABASE_URI")
if not SUPABASE_DB_URI:
    raise RuntimeError("DATABASE_URI is not set. Please add it to your .env file.")

engine = create_engine(SUPABASE_DB_URI)
db = SQLDatabase(engine)
sql_tool = QuerySQLDatabaseTool(db=db)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    timeout=60
)

# ---------------- QDRANT CONFIG ----------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 1536))

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )


# ---------------- STATE ----------------
class AgentState(TypedDict):
    question: str
    company_name: str | None
    answer: str
    history: list[dict]
    intent: str | None
    intent_stage: str | None


BOTIVATE_TROUBLESHOOT_PROMPT = """
Botivate AI Consultant
(Customer Portal – Master Prompt | Locked Version)
🧠 IDENTITY & ROLE
You are Botivate AI Consultant — a world-class senior business transformation, operations, systems, Google Workspace, and technical troubleshooting consultant.
You work inside the Botivate Customer Portal.
Your responsibility is to:
Understand customer problems deeply and empathetically
Solve operational & technical issues step-by-step
Identify which Botivate system can solve the problem
Guide customers on data collection & SOP clarity
Help with Google Sheets, Google Workspace, Looker Studio issues
Handle team resistance and system adoption
Escalate to Botivate human team via ticket when execution is needed
Your mindset:
Solve first → Clarify second → System-fit third → Execute via ticket
🌍 LANGUAGE RULE (ABSOLUTE)
Always reply in the same language as the user
Hindi → Hindi
Hinglish → Hinglish
English → English
Never switch language yourself.
🗣️ TONE & FEEL
Human
Calm
Respectful
Consultant-like
Simple & practical
Never salesy
Never robotic
Customer should feel:
“Ye meri problem samajh raha hai aur practical guidance de raha hai.”
🧠 CORE THINKING MODEL
Every problem belongs to one (or more):
System missing
System weak
People not aligned
Owner lacks visibility
Technical / data error
Your job is to identify the real root cause, not just symptoms.
🔁 UNIVERSAL RESPONSE FLOW (MANDATORY)
1️⃣ EMPATHY FIRST (1–2 lines)
2️⃣ QUICK DIAGNOSIS (1 line)
3️⃣ STEP-BY-STEP SOLUTION (MAX 6 STEPS)
4️⃣ CONFIRM STATUS
5️⃣ SYSTEM-FIT EXPLANATION (ONLY WHEN SYSTEM IS NEEDED)
6️⃣ HUMAN HANDOVER (ONLY IF USER EXPLICITLY ASKS FOR IMPLEMENTATION)

🔹 SYSTEM DESIGN MODE (VERY IMPORTANT)
- Jab user bole: "workflow design karo", "system design chahiye", "policy banana hai"
  → Pehle 3–4 follow-up questions puchho (process, team, approvals, frequency)
  → Phir unke jawab ke basis par clear, practical workflow draft karo
  → Sirf tab handover + ticket suggest karo jab user clearly bole:
    "ab aap hi implement / setup kar do" ya "Botivate team implement kare"

🎫 TICKET RULE (STRICT)
Suggest ticket only when:
- Troubleshooting attempt ho chuka hai, YA
- User clearly implementation / setup maang raha ho
Tab tone:
“Is problem ke liye system setup Botivate team karegi. Aap apna data aur process points ready kar lijiye aur please customer portal me ticket raise kar dijiye. Hamari team aapko call karke system design aur implement kar degi.”
"""

LANGUAGE_CONTROLLER_PROMPT = """
You must strictly follow the user's language.

Rules:
- If the user's message is in English → respond ONLY in English.
- If the user's message is in Hinglish → respond ONLY in Hinglish.
- If the user's message is in Hindi → respond ONLY in Hindi.
- Do NOT mix languages.
- Do NOT translate unless the user explicitly asks.
- Follow the user's language even if other instructions differ.
"""

def is_new_conversation(history: list[dict]) -> bool:
    return not history or len(history) == 0


def get_system_prompt(history: list[dict]) -> str:
    base_prompt = BOTIVATE_TROUBLESHOOT_PROMPT + "\n\n" + LANGUAGE_CONTROLLER_PROMPT
    if is_new_conversation(history) and not any(
        h["role"] == "assistant" for h in history
    ):
        return (
            base_prompt
            + "\nCONVERSATION STATE:\n- This is the FIRST message of the conversation.\n"
            "- You may greet the user briefly and naturally."
        )
    return (
        base_prompt
        + "\nCONVERSATION STATE:\n- This is an ONGOING conversation.\n"
        "- DO NOT greet again.\n- Continue naturally from previous context."
    )


def embed_text(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL"),
        input=text
    )
    return response.data[0].embedding

def store_memory(question: str, answer: str, intent: str, company: str | None):
    # ❌ Do not store implementation / ticket responses
    lowered = answer.lower()
    if "ticket raise" in lowered or "implementation karegi" in lowered:
        return

    vector = embed_text(question)

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "question": question,
                    "answer": answer,
                    "intent": intent,
                    "company": company,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        ]
    )

def store_memory_async(question, answer, intent, company):
    threading.Thread(
        target=store_memory,
        args=(question, answer, intent, company),
        daemon=True
    ).start()


def search_similar_conversations(question: str, recent_history: list, company: str | None, top_k=3):
    """Current + past conversation search (ChatGPT-jaisa)."""
    recent_texts = [msg["content"] for msg in recent_history[-4:] if msg["role"] == "human"]
    full_query = f"{question}\nRecent: {' | '.join(recent_texts[-2:])}" if recent_texts else question
    vector = embed_text(full_query)

    result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
        score_threshold=0.65,
        with_payload=True
    )

    contexts = []
    for point in result.points or []:
        if point.score and point.score >= 0.65:
            payload = point.payload or {}
            if not company or payload.get("company") == company:
                contexts.append(
                    {
                        "question": payload.get("question", ""),
                        "answer": payload.get("answer", ""),
                        "score": point.score,
                        "intent": payload.get("intent"),
                    }
                )
    return contexts[:2]


def get_context_aware_prompt(history, contexts):
    base_prompt = get_system_prompt(history)
    context_str = "\n🧠 SIMILAR PAST CONVERSATIONS:\n"
    for i, ctx in enumerate(contexts, 1):
        context_str += (
            f"Q{i}: {ctx['question'][:100]}...\n"
            f"A{i}: {ctx['answer'][:150]}...\n"
            f"[Score: {ctx['score']:.2f}, Intent: {ctx.get('intent', 'UNKNOWN')}]\n\n"
        )
    return (
        base_prompt
        + f"\n{context_str}\nUse the above similar conversations ONLY as reference."
        "\nAlways answer for the CURRENT question and do not blindly repeat past handover / ticket messages."
    )


def build_conversation_prompt(system_prompt, history, user_question):
    messages = [("system", system_prompt)]
    for h in history[-5:]:
        messages.append((h["role"], h["content"]))
    messages.append(("human", user_question))
    return ChatPromptTemplate.from_messages(messages)


# ---------------- SQL HELPERS ----------------
def generate_sql(question: str, company: str) -> str:
    system_prompt = f"""
You are an expert PostgreSQL assistant.
Table name: "FMS" (case-sensitive, always double quotes).
MANDATORY: Always include WHERE party_name = '{company}'.
Columns: party_name, actual3, planned3, task_no, description_of_work.
TASK STATUS: Completed → actual3 IS NOT NULL, Pending → actual3 IS NULL.
Use COUNT(*) for counts, LIMIT 5 by default if no limit given.
ONLY return raw SQL, no explanation, no markdown.

Schema:
{db.get_table_info()}
"""
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{question}")]
    )
    return (prompt | llm).invoke({"question": question}).content.strip()


def run_sql_and_summarize(question: str, sql: str) -> str:
    try:
        raw_result = sql_tool.invoke({"query": sql})

        # 🔐 Normalize SQL tool output
        if isinstance(raw_result, str):
            result = eval(raw_result)   # SQL tool returns Python-style string
        else:
            result = raw_result
    except Exception as e:
        print("SQL ERROR:", e)
        return "I’m unable to fetch data right now. Please try again."

    if not result or str(result).strip() in ["[]", "None", ""]:
        return "There are no matching tasks."

    q = question.lower()
    is_count_query = any(
        word in q for word in ["how many", "count", "total"]
    )

    # ---------- COUNT ----------
    if is_count_query:
        count = 0
        row = result[0]

        if isinstance(row, dict):
            count = int(row.get("task_count", 0))
        elif isinstance(row, (list, tuple)):
            count = int(row[0])
        else:
            count = int(row)

        if "completed" in q:
            return f"Completed tasks: {count}"
        if "pending" in q:
            return f"Pending tasks: {count}"
        return f"Total tasks: {count}"

    # ---------- LIST ----------
    summary_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", LANGUAGE_CONTROLLER_PROMPT),
            (
                "human",
                f"""
User Question: {question}
Database Result: {result}

Rules:
- List tasks only
- One task per line
- Format exactly as:
  Task <task_no>: <description_of_work>
- Do NOT ask follow-up questions
"""
            ),
        ]
    )
    return (summary_prompt | llm).invoke({}).content.strip()
    
def simple_classify(q: str, company: str | None, history: list[dict]) -> str:
    q_lower = q.lower()

    # 🎫 TICKET INTENT (TOP PRIORITY)
    if any(word in q_lower for word in [
        "raise ticket",
        "generate ticket",
        "create ticket",
        "submit ticket",
        "ticket"
    ]):
        return "TICKET"

    # 🔒 CONTEXT LOCK: System design
    if any(
        h["role"] == "assistant" and "attendance system" in h["content"].lower()
        for h in history[-4:]
    ):
        return "SYSTEM_DESIGN"

    # SQL
    if any(word in q_lower for word in [
        "task", "tasks",
        "total", "completed", "pending",
        "how many", "count", "number of",
        "give me", "show", "list"
    ]):
        return "SQL"

    # SYSTEM DESIGN
    if any(word in q_lower for word in [
        "workflow", "system design", "policy",
        "attendance system", "implement"
    ]):
        return "SYSTEM_DESIGN"

    return "CHAT"

def decide_system_stage(history: list[dict]) -> str:
    """
    Uses LLM to decide whether:
    - DISCOVERY is still needed
    - or WORKFLOW can be generated
    """

    judge_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
You are a senior business consultant.

Your task:
Analyze the conversation and decide whether there is
ENOUGH information to design a COMPLETE attendance system workflow.

Return ONLY one word:
- DISCOVERY  → if more clarification is needed
- WORKFLOW   → if workflow can be designed now

Rules:
- Do NOT ask questions
- Do NOT explain
- Judge based on practical business sense
"""),
            ("human", """
Conversation history:
{history}
""")
        ]
    )

    response = (judge_prompt | llm).invoke(
        {"history": json.dumps(history, indent=2)}
    ).content.strip().upper()

    return "WORKFLOW" if "WORKFLOW" in response else "DISCOVERY"

def decide_conversation_action(history: list[dict], question: str) -> str:
    """
    Let LLM decide what the user wants NEXT.
    """
    decision_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
You are an expert conversation state analyzer.

Based on the conversation so far, decide what the user expects NEXT.

Return ONLY ONE word from:
- DISCOVERY        (needs clarification questions)
- WORKFLOW         (needs system/workflow design)
- IMPLEMENTATION   (needs implementation steps of the LAST workflow)
- TICKET           (wants to raise/generate a ticket)
- SQL              (data / task query)
- CHAT             (general conversation)

Rules:
- Use conversation history + current message
- If user says "this", infer context from history
- Do NOT explain
- Do NOT ask questions
"""),
            ("human", """
Conversation history:
{history}

User message:
{question}
""")
        ]
    )

    result = (decision_prompt | llm).invoke(
        {
            "history": json.dumps(history, indent=2),
            "question": question
        }
    ).content.strip().upper()

    return result

def is_system_design_active(history: list[dict]) -> bool:
    """
    Returns True if the conversation is already
    in workflow / system design mode.
    """
    for h in reversed(history[-6:]):
        if h["role"] == "assistant":
            text = h["content"].lower()
            if any(word in text for word in [
                "workflow",
                "process design",
                "system design",
                "follow-up questions",
                "purchase process"
            ]):
                return True
    return False

def has_already_asked_questions(history: list[dict]) -> bool:
    """
    Returns True if assistant already asked discovery questions.
    """
    for h in history:
        if h["role"] == "assistant":
            if "?" in h["content"]:
                return True
    return False

def handle_conversation(state: AgentState):
    q = state["question"]

    # 🔧 NORMALIZE USER INPUT (typo tolerance)
    q = q.lower().replace("taks", "tasks").strip()

    company = state.get("company_name")
    history = state["history"]

    # ✅ SINGLE SOURCE OF TRUTH FOR INTENT
    intent = decide_conversation_action(history[-6:], q)

    # 🔐 HARD SQL LOCK (ONLY ONCE)
    if any(word in q for word in ["task", "tasks"]):
        intent = "SQL"

    # 🔒 CONTEXT-FIRST INTENT DECISION

    # if is_system_design_active(history):
    #     intent = "SYSTEM_DESIGN"
    # else:
    #     intent = decide_conversation_action(
    #         history[-6:],
    #         q
    #     )
    # # 🔒 HARD SQL OVERRIDE
    # if any(word in q.lower() for word in [
    #     "task", "tasks", "give me", "show", "list"
    # ]):
    #     intent = "SQL"

    # # 🔒 HARD CONTEXT LOCK FOR WORKFLOW DISCOVERY
    # if any(
    #     h["role"] == "assistant"
    #     and "workflow" in h["content"].lower()
    #     for h in history[-4:]
    # ):
    #     intent = "SYSTEM_DESIGN"

    contexts = []
    if len(history) >= 2:
        contexts = search_similar_conversations(q, history, company)

    system_prompt = (
        get_context_aware_prompt(history, contexts)
        if contexts
        else get_system_prompt(history)
    )

    # ---------- SQL ----------
    if intent == "SQL":
        if not company:
            answer = "Company context missing. Please refresh and try again."
            return {"answer": answer, "history": history}
        limit = extract_limit(q)
        try:
            q_lower = q.lower()
            is_count_query = any(
                phrase in q_lower
                for phrase in ["how many", "total", "count", "number"]
            )

            if is_count_query:
                if "completed" in q_lower:
                    sql = f'''
                    SELECT COUNT(*) AS task_count
                    FROM "FMS"
                    WHERE TRIM(party_name) ILIKE TRIM('{company}')
                    AND actual3 IS NOT NULL
                    '''
                elif "pending" in q_lower:
                    sql = f'''
                    SELECT COUNT(*) AS task_count
                    FROM "FMS"
                    WHERE TRIM(party_name) ILIKE TRIM('{company}')
                    AND actual3 IS NULL
                    '''
                else:
                    sql = f'''
                    SELECT COUNT(*) AS task_count
                    FROM "FMS"
                    WHERE TRIM(party_name) ILIKE TRIM('{company}')
                    '''
            else:
                if "completed" in q_lower:
                    sql = f'''
                    SELECT task_no, description_of_work
                    FROM "FMS"
                    WHERE TRIM(party_name) ILIKE TRIM('{company}')
                    AND actual3 IS NOT NULL
                    ORDER BY task_no DESC
                    LIMIT {limit}
                    '''
                elif "pending" in q_lower:
                    sql = f'''
                    SELECT task_no, description_of_work
                    FROM "FMS"
                    WHERE TRIM(party_name) ILIKE TRIM('{company}')
                    AND actual3 IS NULL
                    ORDER BY task_no DESC
                    LIMIT {limit}
                    '''
                else:
                    sql = f'''
                    SELECT task_no, description_of_work
                    FROM "FMS"
                    WHERE TRIM(party_name) ILIKE TRIM('{company}')
                    ORDER BY task_no DESC
                    LIMIT {limit}
                    '''
            answer = run_sql_and_summarize(q, sql)
        except Exception:
            answer = "I’m unable to fetch data right now. Please try again."

        store_memory_async(q, answer, "SQL", company)

        history.append({"role": "human", "content": q})
        history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "history": history
        }

    # ---------- TICKET ----------
    if intent == "TICKET":
        ticket_prompt = system_prompt + """
    The user wants to generate a ticket.

    STRICT RULES:
    - Do NOT ask open-ended questions
    - Do NOT greet again
    - Do NOT act like chat
    - Guide the user to raise a ticket from the Dashboard → Generate Ticket page

    Explain clearly:
    1. What to select in Date
    2. What to enter in Person Name
    3. How to add main task
    4. How to add additional tasks
    5. What details are important for faster resolution

    IMPORTANT:
    - Do NOT say "customer portal"
    - Always say: "Dashboard ke Generate Ticket page se"

    End the response with a SEPARATE LINE:
    "➡️ Ticket raise karne ke liye Dashboard ke Generate Ticket page par ja kr Ticket raise kr skte hai... Ticket raise krne se related koi bhi query ho to aap mujhse puch skte hai."
    """

        history.append({"role": "human", "content": q})

        prompt = build_conversation_prompt(
            ticket_prompt,
            history,
            q
        )


        answer = (prompt | llm).invoke({}).content.strip()

        history.append({"role": "assistant", "content": answer})


        return {
            "answer": answer,
            "history": history
        }

    if intent == "IMPLEMENTATION":
        implementation_prompt = system_prompt + """
    The user is asking how to IMPLEMENT the system that was discussed most recently.

    Rules:
    - Infer "this" from conversation history
    - Do NOT greet
    - Do NOT ask questions
    - Do NOT redesign the workflow
    - Give clear, practical implementation steps
    - Mention tools, setup, ownership, and sequence
    """

        prompt = build_conversation_prompt(
            implementation_prompt,
            history,
            q
        )

        answer = (prompt | llm).invoke({}).content.strip()

        history.append({"role": "human", "content": q})
        history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "history": history
        }
    # ---------- SYSTEM DESIGN ----------
    if intent == "SYSTEM_DESIGN":

        # ⛔ Stop repeated discovery questioning
        if has_already_asked_questions(history):
            workflow_prompt = system_prompt + """
You are now in FINAL WORKFLOW MODE.
Do NOT ask questions.
Make reasonable assumptions if required.
"""
        else:
            workflow_prompt = system_prompt + """
You may ask at most 2 clarification questions.
Then move directly to workflow.
"""

        prompt = build_conversation_prompt(workflow_prompt, history, q)
        answer = (prompt | llm).invoke({}).content.strip()

        # ✅ FORCE TICKET CTA AFTER FINAL WORKFLOW
        if any(phrase in answer.lower() for phrase in [
            "koi changes chahiye",
            "aur koi madad chahiye",
            "any changes",
            "anything else"
        ]):
            answer += (
                "\n\n Agar aap chahte hain ki Botivate team is system ko implement kare, "
                "to Dashboard ke Generate Ticket page se ticket raise kar sakte hain."
            )

        store_memory_async(q, answer, "SYSTEM_DESIGN_FINAL", company)

        history.append({"role": "human", "content": q})
        history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "history": history
        }

    # ---------- CHAT / TROUBLESHOOT ----------
    prompt = build_conversation_prompt(system_prompt, history, q)
    answer = (prompt | llm).invoke({}).content.strip()

    store_memory_async(q, answer, intent or "CHAT", company)

    history.append({"role": "human", "content": q})
    history.append({"role": "assistant", "content": answer})

    return {
        "answer": answer,
        "history": history
    }


# ---------------- GRAPH ----------------
graph = StateGraph(AgentState)
graph.add_node("conversation", handle_conversation)
graph.set_entry_point("conversation")
graph.add_edge("conversation", END)
agent = graph.compile()