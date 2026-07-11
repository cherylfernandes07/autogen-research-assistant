# 🔬 AutoGen Research Assistant

A multi-agent AI system that automates end-to-end academic research workflows — from topic refinement to gap analysis — using AutoGen, the arXiv API, and your choice of LLM provider.

## Overview

Manual literature reviews are slow and inconsistent. This project orchestrates five specialized AI agents that collaborate to handle the full research pipeline: refining broad topics into precise questions, discovering relevant papers via arXiv, extracting key insights, compiling structured reports, and identifying research gaps — with a human-in-the-loop approval checkpoint between discovery and analysis.

Built as a demonstration of multi-agent orchestration patterns using AutoGen's sequential workflow, tool registration, caller/executor separation, custom selector functions, and HITL design.

## Demo

```
Topic: "The impact of microplastics on marine life."

→ TopicRefinementAgent    refines into a precise research question + keywords
→ PaperDiscoveryAgent     queries arXiv, returns 5 relevant papers
                          ↳ custom selector stops pipeline on PAPERS_FOUND
→ [HITL checkpoint]       human reviews and approves paper list
→ InsightSynthesizerAgent extracts 5 key findings with paper citations
→ ReportCompilerAgent     compiles a 4-section structured report
→ GapAnalysisAgent        identifies gaps + 3 future research directions
→ Pipeline terminates cleanly on GAP_ANALYSIS_COMPLETE
```

## Architecture

```
User Input (broad topic)
        │
        ▼
┌─────────────────────┐
│ Topic Refinement    │  Sharpens query into precise research scope + keywords
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Paper Discovery     │  Calls arXiv API via registered tool
│                     │  ← custom selector stops on PAPERS_FOUND signal
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ User Proxy (HITL)   │  Human approval checkpoint — type 'approve' to proceed
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Insight Synthesizer │  Extracts key findings with paper citations
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Report Compiler     │  Structures findings into a 4-section report
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Gap Analysis        │  Identifies open questions + future directions
└─────────────────────┘
```

## Features

- **5-agent sequential pipeline** orchestrated via AutoGen's `initiate_chats`
- **arXiv integration** for real-time academic paper retrieval via registered tool
- **Caller/executor pattern** — LLM decides when to call tools, UserProxy executes them safely
- **Custom selector function** — stops paper discovery immediately on `PAPERS_FOUND`, no wasted turns
- **Context chaining** — each agent's output is automatically passed to the next via `summary_method`
- **Human-in-the-loop approval** — pipeline pauses after paper discovery for human review
- **Provider-agnostic LLM config** — swap between Gemini, OpenAI, or Groq in one line
- **Termination conditions** to prevent runaway agent loops

## How It Works — Custom Selector Function

One of the more interesting architectural challenges in this project was controlling *when* the `PaperDiscoveryAgent` should stop talking. AutoGen's default behaviour relies on `max_turns` — a hard ceiling on the number of exchanges — but this meant the agent would keep repeating its paper summaries until it hit the limit, wasting turns and adding noise to the output. The fix was to implement a custom reply function registered directly on the `UserProxy` agent, triggered specifically when it's in conversation with `PaperDiscoveryAgent`.

The selector reads the last message after every exchange and checks for the `PAPERS_FOUND` signal. The moment it detects it, it returns `(True, None)` — AutoGen's convention for "stop and return nothing" — cutting the conversation short immediately. This means the agent calls the arXiv tool once, summarises the results, says `PAPERS_FOUND`, and the pipeline moves on without a single wasted turn. It's a small change in code but a meaningful shift in how the workflow is controlled: rather than relying on turn limits as a blunt instrument, the system now responds to the *semantic state* of the conversation.

## Tech Stack

- [AutoGen](https://github.com/microsoft/autogen) — multi-agent orchestration framework
- [Google Gemini](https://ai.google.dev/) (`gemini-2.0-flash`) — default LLM backbone
- [Groq](https://console.groq.com/) (`llama-3.3-70b-versatile`) — fast, rate-limit-friendly alternative
- [arXiv API](https://arxiv.org/help/api/) — academic paper search
- Python 3.9+

## Getting Started

### Prerequisites

- Python 3.9 or higher
- An API key for at least one provider:
  - [Google AI Studio](https://aistudio.google.com/) — Gemini (free tier works)
  - [Groq Console](https://console.groq.com/) — Groq (generous free tier, good for dev)
  - [OpenAI Platform](https://platform.openai.com/) — OpenAI

### Installation

```bash
git clone https://github.com/your-username/autogen-research-assistant.git
cd autogen-research-assistant

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your-gemini-api-key-here
GROQ_API_KEY=your-groq-api-key-here

# Optional
OPENAI_API_KEY=your-openai-key
```

### Run

```bash
python main.py
```

When prompted at the HITL checkpoint, type `approve` to continue, or describe changes to redirect the research.

To change the research topic, edit `broad_topic` in `main.py`:

```python
broad_topic = "The impact of microplastics on marine life."
```

To switch LLM providers, change `PROVIDER`:

```python
PROVIDER = "gemini"  # options: "gemini", "groq", "openai"
```

> **Tip:** If you hit Gemini rate limits during development, switch to `"groq"` — it has a much more generous free tier for experimentation.

## Project Structure

```
autogen-research-assistant/
├── main.py          # Agent definitions, tool registration, custom selector, workflow
├── aimodels.py      # Provider-agnostic LLM config (Gemini, OpenAI, Groq)
├── requirements.txt
├── .env             # API keys (not committed)
└── .gitignore
```

## Status

| Feature | Status |
|---|---|
| Topic Refinement Agent | ✅ Complete |
| Paper Discovery Agent + arXiv tool | ✅ Complete |
| Insight Synthesizer Agent | ✅ Complete |
| Report Compiler Agent | ✅ Complete |
| Gap Analysis Agent | ✅ Complete |
| Full sequential pipeline (end-to-end) | ✅ Complete |
| Human-in-the-loop (HITL) checkpoint | ✅ Complete |
| Custom selector function | ✅ Complete |
| Provider-agnostic LLM config | ✅ Complete |
| Export report to markdown file | 🔜 Planned |

## Known Limitations

- **arXiv search quality** — results depend on how well the TopicRefinementAgent constructs keywords. Occasionally off-topic papers appear. Prompt tuning or a post-filter step would improve precision.
- **Python 3.9** — several Google dependencies warn about end-of-life support. Upgrading to Python 3.10+ is recommended for production use.

## Roadmap

- [x] 5-agent sequential pipeline running end-to-end
- [x] arXiv tool with caller/executor registration pattern
- [x] Custom selector function for strict workflow routing
- [x] Human-in-the-loop approval checkpoint
- [x] Multi-provider LLM support (Gemini, Groq, OpenAI)
- [ ] Export final report to a `.md` file
- [ ] Add support for PubMed and Semantic Scholar APIs
- [ ] Upgrade to Python 3.10+

## Author

**Cheryl Fernandes**  
AI Engineer · Founder, Nirvaan Labs  
[LinkedIn](https://www.linkedin.com/in/cheryl-fernandes-90967611/) · [GitHub](https://github.com/cherylfernandes07)