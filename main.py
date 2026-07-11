import os
import arxiv
import autogen
from aimodels import get_autogen_config

# ============================================================
# CONFIGURATION
# Control your LLM provider here. Options: "openai", "groq", "gemini"
# The get_autogen_config function in aimodels.py handles the rest.
# ============================================================
PROVIDER = "groq"
llm_config = get_autogen_config(PROVIDER)


# ============================================================
# TOOL DEFINITION
# This is a plain Python function — not an agent.
# AutoGen will register it so the PaperDiscoveryAgent can call it
# during the workflow. The UserProxy actually executes it.
# ============================================================
def search_arxiv(query: str, max_results: int = 3) -> str:
    """
    Search arXiv for academic papers given a query string.

    Args:
        query: The search string (e.g. "microplastic bioaccumulation marine")
        max_results: How many papers to return (default 3 to keep output concise)

    Returns:
        A formatted string of paper titles, authors, URLs, and abstracts.
        Returns "No papers found." if the query yields nothing.
    """
    # arxiv.Client() is the modern way to query arXiv (replaces Search.results())
    client = arxiv.Client()

    # Define the search: what to look for, how many results, ranked by relevance
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    for result in client.results(search):
        # Format each paper as a readable block
        # We truncate abstracts to 400 chars to keep the agent's context lean
        papers.append(
            f"Title: {result.title}\n"
            f"Authors: {', '.join([a.name for a in result.authors])}\n"
            f"URL: {result.entry_id}\n"
            f"Abstract: {result.summary[:400]}...\n"
            f"---"
        )

    return "\n".join(papers) if papers else "No papers found."


# ============================================================
# AGENT DEFINITIONS
# Each agent has a specific role defined by its system_message.
# All LLM-powered agents share the same llm_config (same model/key).
# The UserProxy is the only agent with llm_config=False — it
# represents the human/executor side and never calls the LLM itself.
# ============================================================

# --- UserProxy ---
# Acts as the workflow coordinator and tool executor.
# human_input_mode="NEVER" means it runs fully automatically.
# is_termination_msg tells it when the whole pipeline is done.
# We check for GAP_ANALYSIS_COMPLETE because that's the last agent's signal.
user_proxy = autogen.ConversableAgent(
    name="UserProxy",
    llm_config=False,
    human_input_mode="NEVER",
    is_termination_msg=lambda x: "GAP_ANALYSIS_COMPLETE" in (x.get("content") or ""),
    max_consecutive_auto_reply=10
)

# --- Topic Refinement Agent ---
# Takes a broad topic and sharpens it into a precise research question.
# This gives PaperDiscoveryAgent much better search keywords to work with.
# The TOPIC_REFINED signal tells the workflow this stage is complete.
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

# --- Paper Discovery Agent ---
# Uses the refined keywords to search arXiv via the search_arxiv tool.
# Instructed to call the tool ONCE to avoid burning through max_turns.
# After getting results, it summarizes and signals PAPERS_FOUND.
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

# --- Insight Synthesizer Agent ---
# Reads all the paper abstracts and distills the most important findings.
# It doesn't call any tools — pure LLM reasoning over the context passed in.
# Each insight is tied back to a specific paper for traceability.
insight_synthesizer_agent = autogen.ConversableAgent(
    name="InsightSynthesizerAgent",
    llm_config=llm_config,
    system_message="""You are a research insight specialist.
    Given a list of academic papers with titles and abstracts,
    extract the 3-5 most important findings and themes across all papers.
    Present each insight as a clear, concise bullet point with a reference
    to which paper it came from.
    End your response with INSIGHTS_EXTRACTED."""
)

# --- Report Compiler Agent ---
# Takes everything gathered so far and structures it into a readable report.
# The four-section format (Question, Papers, Findings, Conclusion) makes
# the output immediately useful and easy to share.
report_compiler_agent = autogen.ConversableAgent(
    name="ReportCompilerAgent",
    llm_config=llm_config,
    system_message="""You are a research report specialist.
    Given a research question, a list of papers, and extracted insights,
    compile a structured research report with the following sections:
    1. Research Question
    2. Key Papers (title + one line summary each)
    3. Key Findings (synthesized insights)
    4. Conclusion (2-3 sentences)
    End your response with REPORT_COMPILED."""
)

# --- Gap Analysis Agent ---
# The final agent. Reads the compiled report and identifies what's missing
# in the existing literature — unanswered questions, methodological gaps,
# and concrete suggestions for where future research should go.
gap_analysis_agent = autogen.ConversableAgent(
    name="GapAnalysisAgent",
    llm_config=llm_config,
    system_message="""You are a research gap analysis specialist.
    Given a compiled research report, identify:
    1. What questions remain unanswered in the literature
    2. What methodologies are missing or underused
    3. 3 specific suggestions for future research directions
    Be specific and grounded in what the papers actually covered.
    End your response with GAP_ANALYSIS_COMPLETE."""
)


# ============================================================
# TOOL REGISTRATION
# This wires search_arxiv into the AutoGen system.
# caller=paper_discovery_agent means that agent can suggest calling it.
# executor=user_proxy means the UserProxy actually runs the Python function.
# This separation is intentional: the LLM decides WHEN to call the tool,
# but the Python runtime (via UserProxy) actually executes it safely.
# ============================================================
autogen.register_function(
    search_arxiv,
    caller=paper_discovery_agent,
    executor=user_proxy,
    name="search_arxiv",
    description="Search arXiv for academic papers given a query string"
)


# ============================================================
# WORKFLOW EXECUTION
# initiate_chats() runs a sequential pipeline where each chat's
# final message (summary_method="last_msg") is automatically injected
# as context into the next chat's opening message.
# This is how each agent "sees" what the previous one did.
# ============================================================
if __name__ == "__main__":
    broad_topic = "The impact of microplastics on marine life."

    chat_results = user_proxy.initiate_chats([
        {
            # Stage 1: Sharpen the topic into a precise research question
            "recipient": topic_refinement_agent,
            "message": f"Please refine this topic: {broad_topic}",
            "max_turns": 1,
            "summary_method": "last_msg"  # passes refined question to next stage
        },
        {
            # Stage 2: Search arXiv using the refined keywords
            # max_turns=6 gives room for: tool call + result + summary
            "recipient": paper_discovery_agent,
            "message": "Use the refined keywords to find academic papers now.",
            "max_turns": 6,
            "summary_method": "last_msg"  # passes paper list to next stage
        },
        {
            # Stage 3: Extract key insights from the papers
            "recipient": insight_synthesizer_agent,
            "message": "Extract the key insights from these papers.",
            "max_turns": 2,
            "summary_method": "last_msg"  # passes insights to next stage
        },
        {
            # Stage 4: Compile everything into a structured report
            "recipient": report_compiler_agent,
            "message": "Compile a structured research report from everything gathered so far.",
            "max_turns": 2,
            "summary_method": "last_msg"  # passes report to next stage
        },
        {
            # Stage 5: Identify gaps and future research directions
            "recipient": gap_analysis_agent,
            "message": "Identify the research gaps and suggest future directions.",
            "max_turns": 2,
            "summary_method": "last_msg"
        }
    ])