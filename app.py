import streamlit as st
import time
from main import agent_executor, parser

st.set_page_config(page_title="AI Research", page_icon="🔍", layout="centered")

st.markdown("""
    <style>
    .main .block-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 80vh;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🛠️ Kontrol Panel")
    if st.button("Hapus Riwayat Chat 🧹"):
        st.session_state.messages = []
        st.rerun()

st.markdown("<h1 style='text-align: center;'> AI Research</h1>", unsafe_allow_html=True)

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Apa yang ingin anda riset hari ini?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Currently Researching....", expanded=True) as status:
            try:
                full_query = f"""
                Research this topic thoroughly:
                {prompt}
                IMPORTANT:
                - Use the SAME language as the user's question.
                - If the question is Indonesian, answer in Indonesian.
                - If the question is English, answer in English.
                - Do not switch languages unless requested.
                Requirements:
                - Minimum 500 words
                - Detailed explanation
                - Multiple paragraphs
                - Include background information
                - Include examples when relevant
                - Explain every important point
                - Cite sources

                Return ONLY valid JSON.

                {parser.get_format_instructions()}
                """
                response = agent_executor.invoke({"messages": [("human", full_query)]})
                
                last_message_obj = response["messages"][-1]
                if isinstance(last_message_obj.content, list):
                    last_msg = last_message_obj.content[0].get('text', '')
                else:
                    last_msg = last_message_obj.content
                start = last_msg.find('{')
                end = last_msg.rfind('}') + 1
                clean_json = last_msg[start:end]
                data = parser.parse(clean_json)
                
                status.update(
                    label="The Research is Complete! ✅",
                    state="complete"
                )
                summary = data.summary.replace("```", "").strip()

                result_text = f"## 💡 Hasil Penelitian: {data.topic}\n\n"
                result_text += "## Ringkasan\n\n"
                result_text += f"{summary}\n\n"
                result_text += "## Referensi\n\n"

                for src in data.sources:
                    result_text += f"- {src}\n"

                st.markdown(result_text) 
                st.session_state.messages.append({"role": "assistant", "content": result_text})
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat riset: {e}")
                status.update(label="Gagal ❌", state="error")