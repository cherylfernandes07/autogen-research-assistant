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
def search_arxiv(query: str) -> str:
    """Search arXiv for academic papers given a query string"""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=5,  # 👈 hardcode this instead of accepting as parameter
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

# --- Human Approval Agent (HITL) ---
# This is a SECOND user proxy, separate from the tool executor above.
# human_input_mode="ALWAYS" means it pauses and waits for your input
# every time it receives a message — giving you a chance to review
# the papers found and approve, redirect, or refine before analysis begins.
# Unlike the silent user_proxy above, this one represents YOU in the loop.
human_approval_agent = autogen.UserProxyAgent(
    name="HumanApprovalAgent",
    human_input_mode="ALWAYS",  # pauses for your input every turn
    llm_config=False,
    # Passes control forward once you type "approve" or "looks good"
    code_execution_config=False,
    is_termination_msg=lambda x: any(
        word in (x.get("content") or "").lower()
        for word in ["approve", "looks good", "proceed", "continue"]
    ),
    max_consecutive_auto_reply=0  # never auto-reply — always waits for you
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

    # Stage 1: Topic refinement
    result_1 = user_proxy.initiate_chats([
        {
            "recipient": topic_refinement_agent,
            "message": f"Please refine this topic: {broad_topic}",
            "max_turns": 1,
            "summary_method": "last_msg"
        },
        {
            "recipient": paper_discovery_agent,
            "message": "Use the refined keywords to find academic papers now.",
            "max_turns": 6,
            "summary_method": "last_msg"
        }
    ])

    # Extract the paper list from the last chat's summary
    papers_summary = result_1[-1].summary

    # --- HITL CHECKPOINT ---
    # Pause here and show the human the papers before continuing.
    # This runs as a direct 2-agent chat, not a sequential stage.
    print("\n" + "="*60)
    print("HUMAN APPROVAL CHECKPOINT")
    print("="*60)
    print("Papers found:\n")
    print(papers_summary)
    print("\nType 'approve' to continue, or describe changes to make.")

    # Direct initiate_chat between human_approval_agent and user_proxy
    # This will actually pause and wait for your keyboard input
    approval_result = human_approval_agent.initiate_chat(
        user_proxy,
        message=f"Please review these papers and type 'approve' to proceed:\n\n{papers_summary}",
        max_turns=2
    )

    # Stage 2: Analysis pipeline with papers as context
    user_proxy.initiate_chats([
        {
            "recipient": insight_synthesizer_agent,
            "message": f"Extract the key insights from these papers:\n\n{papers_summary}",
            "max_turns": 1,  # 👈 was 2 — stop after first response
            "summary_method": "last_msg"
        },
        {
            "recipient": report_compiler_agent,
            "message": "Compile a structured research report from everything gathered so far.",
            "max_turns": 1,  # 👈 same here
            "summary_method": "last_msg"
        },
        {
            "recipient": gap_analysis_agent,
            "message": "Identify the research gaps and suggest future directions.",
            "max_turns": 1,  # 👈 and here
            "summary_method": "last_msg"
        }
    ])