# 🔬 AutoGen Research Assistant

A multi-agent AI system that automates academic research workflows — from topic refinement to paper discovery — using AutoGen, the arXiv API, and Google Gemini.

## Overview

Manual literature reviews are slow and inconsistent. This project orchestrates specialized AI agents that collaborate to handle end-to-end research tasks: refining broad topics into precise questions, discovering relevant papers, extracting insights, compiling structured reports, and identifying research gaps.

Built as a demonstration of multi-agent orchestration patterns using AutoGen's sequential workflow and human-in-the-loop (HITL) design.

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
│ Paper Discovery     │  Calls arXiv API to retrieve relevant papers
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ User Proxy (HITL)   │  Human approval checkpoint
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Insight Synthesizer │  Extracts key findings from abstracts
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Report Compiler     │  Structures findings into a cohesive report
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Gap Analysis        │  Identifies open questions and future directions
└─────────────────────┘
```

## Features

- **Multi-agent orchestration** via AutoGen's sequential chat workflow
- **arXiv integration** for real-time academic paper retrieval
- **Human-in-the-loop approval** between discovery and analysis stages
- **Provider-agnostic LLM config** — swap between Gemini, OpenAI, or Groq in one line
- **Custom selector function** for fine-grained workflow control
- **Termination conditions** to prevent runaway agent loops

## Tech Stack

- [AutoGen](https://github.com/microsoft/autogen) — multi-agent orchestration framework
- [Google Gemini](https://ai.google.dev/) (`gemini-2.0-flash`) — LLM backbone
- [arXiv API](https://arxiv.org/help/api/) — academic paper search
- Python 3.10+

## Getting Started

### Prerequisites

- Python 3.10 or higher
- A [Google AI Studio](https://aistudio.google.com/) API key (free tier works)

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

# Optional — only needed if switching providers
OPENAI_API_KEY=your-openai-key
GROQ_API_KEY=your-groq-key
```

### Run

```bash
python main.py
```

To change the research topic, edit the `broad_topic` variable in `main.py`:

```python
broad_topic = "The impact of microplastics on marine life."
```

To switch LLM providers, change the `PROVIDER` variable:

```python
PROVIDER = "gemini"  # options: "gemini", "openai", "groq"
```

## Project Structure

```
autogen-research-assistant/
├── main.py          # Agent definitions, tool registration, workflow execution
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
| Insight Synthesizer Agent | 🔄 In progress |
| Report Compiler Agent | 🔄 In progress |
| Gap Analysis Agent | 🔄 In progress |
| Human-in-the-loop (HITL) | 🔄 In progress |
| Custom selector function | 🔄 In progress |

## Roadmap

- [ ] Add insight synthesis and report compilation agents
- [ ] Implement human-in-the-loop approval checkpoint
- [ ] Build custom selector function for workflow routing
- [ ] Export final report to markdown file
- [ ] Add support for PubMed and Semantic Scholar APIs

## Author

**Cheryl Fernandes**  
AI Engineer · Founder, Nirvaan Labs  
[LinkedIn](https://www.linkedin.com/in/cheryl-fernandes-90967611/) · [GitHub](https://github.com/cherylfernandes07)