from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.prebuilt import create_react_agent
from tools import search_tool
import sys

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=4096, max_retries=3)

class ResearchResponse(BaseModel):
    topic: str = Field(description="The research topic")
    summary: str = Field(description="Detailed research summary")
    sources: list[str] = Field(description="List of sources used")
    tools_used: list[str] = Field(description="List of tools used")

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

system_prompt = """
You are an elite AI Research Assistant. Your mission is to provide accurate, evidence-based, and comprehensive research results.

RESEARCH GUIDELINES:
1. You have access to a 'search_tool'. You MUST use this tool to verify facts, gather data, and retrieve up-to-date information for every request.
2. Cross-check information from multiple sources whenever possible.
3. Provide detailed explanations that include context, history, and key implications.
4. If information cannot be verified, explicitly state the uncertainty.

OUTPUT REQUIREMENTS:
- After performing your research, provide the final answer as a VALID JSON object.
- The JSON must contain these exact keys: "topic", "summary", "sources", "tools_used".
- Do not use markdown (no ```json tags).
- Do not add any conversational text, explanations, or filler after the JSON object.
- If the user asks in Indonesian, answer in Indonesian. If in English, answer in English.
"""

tools = [search_tool]
agent_executor = create_react_agent(llm, tools, prompt=system_prompt)

if __name__ == "__main__":
    query = input("What can I help you research? ")
    
    full_query = (
        f"Research this topic thoroughly: '{query}'. "
        f"Adjust the answer length to the complexity of the question. "
        f"For simple questions, answer concisely but completely. "
        f"For complex topics, provide detailed explanations with context, examples, and analysis. "
        f"{parser.get_format_instructions()}"
    )
    
    inputs = {"messages": [("human", full_query)]}
    
    print("\nBentar Brohh sedang mikir...")
    raw_response = agent_executor.invoke(inputs)
    
    last_message_obj = None
    for msg in reversed(raw_response["messages"]):
        if msg.type == 'ai' and msg.content:
            last_message_obj = msg
            break
            
    if not last_message_obj:
        print("Error: Tidak ada pesan AI yang valid ditemukan.")
        sys.exit()
    
    if isinstance(last_message_obj.content, list):
        last_message_text = "".join([item.get('text', '') if isinstance(item, dict) else str(item) for item in last_message_obj.content])
    else:
        last_message_text = str(last_message_obj.content)

    start = last_message_text.find('{')
    end = last_message_text.rfind('}') + 1

    if start != -1 and end > start:
        clean_json = last_message_text[start:end].strip()
        try:
            structured_response = parser.parse(clean_json)

            with open("research_output.txt", "a", encoding="utf-8") as f:
                f.write(f"Topik: {structured_response.topic}\n")
                f.write(f"Ringkasan: {structured_response.summary}\n")
                f.write(f"Sumber: {', '.join(structured_response.sources)}\n")
                f.write("-" * 20 + "\n") 
            print("\n--- Hasil Penelitian ---")
            print(f"Topik: {structured_response.topic}")
            print(f"Ringkasan: {structured_response.summary[:150]}...") 
            print(f"Sumber: {', '.join(structured_response.sources)}")
            print(f"Tools: {', '.join(structured_response.tools_used)}")

        except Exception as e:
            print("\n--- Gagal Parsing JSON ---")
            print(f"Error: {e}")
            print("Raw JSON yang diproses:", clean_json)
    else:
        print("\nError: Tidak menemukan struktur JSON dalam response.")
        print("Raw response yang diterima:", last_message_text)