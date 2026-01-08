import streamlit as st
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import SerpAPIWrapper

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Red Dragon",
    page_icon="🐉",
    layout="wide"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp { background-color:#0f0f0f; color:#e0e0e0; }
h1 { color:#ff3333; font-family:Courier New; }
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("🔒 System Controls")

    openai_key = st.text_input("OpenAI API Key", type="password")
    serpapi_key = st.text_input("SerpAPI Key (optional)", type="password")

    st.markdown("---")
    with st.expander("Privacy Policy"):
        st.markdown("""
        • Session-only memory  
        • No permanent data storage  
        • APIs used only during request
        """)

    st.markdown("---")
    st.caption("AI Red Dragon v1.0")

# ---------------- AGENT INIT ----------------
@st.cache_resource
def init_agent(api_key, search_key):
    if not api_key:
        return None

    def veo_image(prompt):
        return f"[VEO_IMAGE] {prompt}"

    def veo_video(prompt):
        return f"[VEO_VIDEO] {prompt}"

    tools = [
        Tool(
            name="Search",
            func=SerpAPIWrapper(serpapi_api_key=search_key).run,
            description="Web search"
        ),
        Tool(name="Veo_Image", func=veo_image, description="Image mock"),
        Tool(name="Veo_Video", func=veo_video, description="Video mock"),
    ]

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    llm = ChatOpenAI(
        temperature=0.7,
        model="gpt-3.5-turbo",
        openai_api_key=api_key
    )

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=False
    )

agent = init_agent(openai_key, serpapi_key)

# ---------------- CHAT UI ----------------
st.title("🐉 AI RED DRAGON")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Enter your directive")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if agent:
            with st.spinner("Red Dragon thinking..."):
                reply = agent.run(prompt)
                st.markdown(reply)
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply}
                )
        else:
            st.error("Enter API key to activate AI")
