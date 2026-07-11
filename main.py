import os 
import arxiv 
import autogen 
from aimodels import get_autogen_config

# Control your provider here: "openai", "groq", or "gemini"
PROVIDER = "gemini" 

# 1. Fetch the correct AutoGen configuration format
llm_config = get_autogen_config(PROVIDER)

# --- 2. Define the Tool ---
def search_arxiv(query: str, max_results: int = 3) -> str:
    """Search arXiv for academic papers given a query string"""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = []
    for result in client.results(search):
        papers.append(
            f"Title: {result.title}\n"
            f"Authors: {', '.join([a.name for a in result.authors])}\n"
            f"URL: {result.entry_id}\n"
            f"Abstract: {result.summary[:400]}...\n"
            f"---"
        )
    return "\n".join(papers) if papers else "No papers found."


# --- 3. Define the Agents ---

# The User Proxy runs the tool behind the scenes and coordinates the flow
user_proxy = autogen.ConversableAgent(
    name="UserProxy",
    llm_config=False, 
    human_input_mode="NEVER",
    # Ends the process once the final goal is met
    is_termination_msg=lambda x: "PAPERS_FOUND" in (x.get("content") or ""),
    max_consecutive_auto_reply=5
)

topic_refinement_agent = autogen.ConversableAgent( 
    name="TopicRefinementAgent", 
    llm_config=llm_config, 
    system_message="""You are a research topic specialist. 
    When given a broad research topic, refine it into a precise, 
    searchable research question. 
    Output a clear 1-2 sentence research scope and 
    3-5 specific search keywords. 
    End your response with TOPIC_REFINED.""" 
)

paper_discovery_agent = autogen.ConversableAgent(
    name="PaperDiscoveryAgent",
    llm_config=llm_config,
    system_message="""You are a paper discovery specialist.
    Using the search keywords provided, call the search_arxiv tool ONCE
    to find relevant papers. Do NOT call the tool more than once.
    After receiving the results, immediately present them with title,
    authors, and a one-sentence summary of each abstract.
    You MUST end your final response with the exact word: PAPERS_FOUND"""
)

# --- 4. Register the Tool Correctly ---
# The PaperDiscoveryAgent is the CALLER (suggests using the tool)
# The UserProxy is the EXECUTOR (actually runs the Python code)
autogen.register_function(
    search_arxiv, 
    caller=paper_discovery_agent, 
    executor=user_proxy, 
    name="search_arxiv", 
    description="Search arXiv for academic papers given a query string" 
)

# --- 5. Run the Workflow (Sequential Chat) ---
if __name__ == "__main__":
    broad_topic = "The impact of microplastics on marine life."
    
    # We initiate a sequential chat to pass the baton from agent to agent
    chat_results = user_proxy.initiate_chats([
        {
            "recipient": topic_refinement_agent,
            "message": f"Please refine this topic: {broad_topic}",
            "max_turns": 1,
            "summary_method": "last_msg"
        },
        {
            "recipient": paper_discovery_agent,
            "message": "Use the refined keywords to find academic papers now.",
            "max_turns": 6,  # 👈 more breathing room
            "summary_method": "last_msg"
        }
    ])