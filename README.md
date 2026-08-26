# Aster & Row Support Agent

An AI-powered customer support agent for **Aster & Row** that combines Retrieval-Augmented Generation (RAG), order lookup tools, conversation memory, source citations, and safety controls to provide grounded customer support without unnecessarily guessing or exposing private information.

---

## Features

- Retrieval-Augmented Generation (RAG)
- Knowledge-base question answering
- Source citations for knowledge-based responses
- Order lookup through a dedicated tool
- Multi-turn conversation handling
- Order ID extraction from natural-language queries
- Handling of unknown/non-existent orders
- Protection against exposing sensitive/internal order information
- Refusal to guess when the knowledge base does not contain sufficient information
- Human-support recommendation when sources conflict or a decision cannot be made safely
- Resistance to untrusted instructions contained in retrieved documents
- Evaluation suite with visible and custom test cases

---

## Architecture

```text
                         ┌──────────────────────┐
                         │      User Query      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      AI Agent        │
                         │   Intent / Routing   │
                         └──────────┬───────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
                 ▼                  ▼                  ▼
          Knowledge Query      Order Lookup      Safety / Privacy
                 │                  │                  │
                 ▼                  ▼                  ▼
          ┌────────────┐      ┌────────────┐     ┌──────────────┐
          │    RAG     │      │   Order    │     │ Refusal /    │
          │  Pipeline  │      │    Tool    │     │ Human Help   │
          └─────┬──────┘      └─────┬──────┘     └──────────────┘
                │                   │
                ▼                   ▼
          ┌────────────┐      ┌────────────┐
          │  ChromaDB  │      │   Order    │
          │ Vector DB   │      │   Data     │
          └─────┬──────┘      └────────────┘
                │
                ▼
          ┌────────────┐
          │   Gemini   │
          │    LLM     │
          └─────┬──────┘
                │
                ▼
       ┌────────────────────┐
       │ Grounded Response  │
       │ + Source Citations │
       └────────────────────┘

```
---
## Setup
1. Clone the repository
  git clone https://github.com/udit2245/Aster-and-Row-Support-Agent.git
  cd Aster-and-Row-Support-Agent

Replace the repository URL above with the final repository URL if the repository name or GitHub username differs.

2. Create a virtual environment
Windows
  python -m venv venv
  venv\Scripts\activate

macOS / Linux
  python3 -m venv venv
  source venv/bin/activate

3. Install dependencies
  pip install -r requirements.txt
Environment Variables

The application requires a Gemini API key.

Create a .env file in the project root:

GEMINI_API_KEY=your_gemini_api_key_here

A safe template is provided in:

.env.example

Running the Agent

After activating the virtual environment and configuring the API key, run:

  python src/main.py

The application can then be used to interact with the support agent.


## Project Structure
```
Aster-and-Row-Support-Agent/
│
├── data/
│   └── knowledge-base documents
│
├── evaluation/
│   ├── visible-cases.json
│   ├── custom-cases.json
│   └── run_evaluation.py
│
├── src/
│   ├── agent/
│   │   ├── __init__.py
│   │   └── agent.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   ├── policy_selector.py
│   │   ├── ranking.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   └── order lookup tools
│   │
│   └── ...
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```
## Retrieval-Augmented Generation

The knowledge-base pipeline follows this general flow:
```
Knowledge Base Documents
          │
          ▼
   Document Loading
          │
          ▼
      Processing
          │
          ▼
     Embeddings
          │
          ▼
      ChromaDB
          │
          │
          ▼
      User Query
          │
          ▼
 Similarity Retrieval
          │
          ▼
 Relevant Context
          │
          ▼
       Gemini
          │
          ▼
 Grounded Response
   + Citations
```
The retrieved knowledge-base content is supplied to the language model as context. This allows the agent to answer company-specific questions using the provided information rather than relying only on the model's general knowledge.

**Order Lookup**

Order-related requests can be routed to an order lookup tool.

For example:

User:
Where is ORD-1007?

The agent can retrieve the relevant order information and return customer-facing details such as order status, carrier, tracking information, or delivery information when available.

The agent can also maintain order context across multiple turns.

Example:

User:
Where is my order?

Agent:
Please provide your order ID.

User:
1007

Agent:
[Looks up ORD-1007 and provides the available order information.]

This demonstrates both multi-turn conversation handling and tool usage.

**Grounding and Safety**

A key design goal of the agent is to avoid confidently inventing information when the available evidence is insufficient.

For example, if the knowledge base does not establish whether a product uses a particular material, the agent should not make an unsupported claim.

Similarly, when two authoritative sources contain conflicting instructions, the agent can acknowledge the conflict and recommend contacting human support instead of choosing an answer arbitrarily.

The agent also avoids exposing internal or sensitive customer information that is not appropriate for the customer-facing response.

**Prompt-Injection Resistance**

The knowledge base may contain text that looks like instructions to the AI system.

The agent treats retrieved documents as information sources, not as unrestricted instructions.

For example, an untrusted document instruction attempting to change a return policy should not override the active policy information used by the agent.

This helps prevent retrieved content from manipulating the agent into ignoring higher-priority system behavior.


## Bug Diary
**Bug 1 — Multi-turn Order ID Resolution**
Reproduction
User:
Where is my order?

Agent:
Please provide your order ID.

User:
1007

The agent initially had difficulty resolving a numeric follow-up to the complete order ID.

Root Cause

Order extraction initially depended too heavily on explicitly formatted identifiers such as:

ORD-1007

and did not consistently resolve numeric references using conversation context.

Fix

The order ID extraction and conversation handling logic was improved so that a numeric follow-up can be resolved against the current conversation context.

Regression Test
User:
Where is ORD-1007?

User:
When will it arrive?

The second message should continue using ORD-1007.

**Bug 2 — Exact Phrase Evaluation**
Reproduction

A response containing:

45-calendar-day return window

was incorrectly marked as failing when the evaluator expected:

45 calendar days

Although the response communicated the intended policy correctly, the evaluator marked it as a failure.

Root Cause

The evaluator relied too heavily on literal phrase matching.

Natural-language responses can express the same concept using different punctuation, word forms, or sentence structures.

Fix

The evaluation logic was made more tolerant of natural-language variation while retaining deterministic checks for important requirements.

Regression Test

The following should be treated as equivalent when the intended requirement is satisfied:

45 calendar days
45-calendar-day return window
45-calendar-day return period

**Bug 3 — Shipping / Policy Wording**
Reproduction

The evaluator could report a missing requirement when the agent communicated the correct shipping policy using wording different from the literal expected phrase.

Root Cause

The evaluator tested whether specific strings existed rather than whether the response satisfied the underlying requirement.

Fix

Evaluation checks were improved so that valid linguistic variations are less likely to produce false failures.

Regression Test

Test variations covering:

Supported destinations
Unsupported destinations
Shipping time
Duties and taxes
International shipping policy

## Known Limitations

The current implementation is a prototype and would require additional work before production deployment.

1. Evaluation Limitations

The deterministic evaluation suite can be overly strict when evaluating natural-language responses.

For example, two responses can communicate the same requirement using different wording.

A production evaluation system should combine deterministic assertions with semantic evaluation.

2. LLM API Dependency

The agent depends on the Gemini API. API availability, latency, quota limits, and pricing can therefore affect the system.

3. Knowledge-base Freshness

The quality of RAG responses depends on the knowledge base being accurate and up to date.

4. Order Data

The current order lookup functionality is designed for the provided project environment rather than a production order-management system.

5. Production Security

A production implementation would require stronger authentication, authorization, rate limiting, monitoring, audit logging, and customer-data protection.

6. Human Escalation

Human handoff is currently represented as a recommendation rather than a fully integrated support-ticket or customer-service workflow.

7. Retrieval Quality

Retrieval performance could be improved further using hybrid search, reranking, better chunking strategies, and more sophisticated retrieval evaluation.

## Evaluation

The repository contains an evaluation suite in the evaluation/ directory.

The evaluation cases cover areas such as:

Knowledge-base retrieval
Required content
Forbidden content
Source grounding
Order lookup
Order ID handling
Multi-turn conversations
Human handoff
Refusal behavior
Sensitive-information handling
Run the Evaluation Suite

From the project root:

python evaluation/run_evaluation.py

The runner loads:

**evaluation/visible-cases.json**
**evaluation/custom-cases.json**
and executes the available cases against the agent.

Evaluation Results
Visible Evaluation Cases

The 15 provided visible cases were manually tested because the deterministic evaluation runner produced false negatives for some valid natural-language responses.

**Manual Result**

15 / 15 visible cases passed

Manual pass rate: 100%
```
Category     	    Cases	      Passed	       Pass Rate
Knowledge / RAG	    5	          5	             100%
Order Lookup	      4	          4	             100%
Multi-turn	        2	          2	             100%
Refusal / Safety	  2	          2	             100%
Policy / Edge Cases	2	          2              100%
Total	              15	        15	           100%
```

## Behaviors Validated
```
-Grounded answers with citations
-Correct order identification and lookup
-Multi-turn order resolution
-Handling of unknown order IDs
-Refusal to guess when information is unavailable
-Sensitive-information refusal
-Prompt-injection resistance
-Policy exceptions
-Conflicting knowledge-base sources
-Human-support recommendation
```
The 100% result above refers to the manual verification of the 15 visible cases, not to the automated evaluator's literal phrase-matching score.

Baseline vs Final

The final implementation was manually validated against all 15 visible cases.

Evaluation	Result
Visible cases manually tested	15
Visible cases passed	15
Manual pass rate	100%

**The automated evaluator was also developed to provide repeatable regression testing. However, deterministic phrase matching can incorrectly reject valid natural-language variations, so the manual visible-case result is reported separately rather than presenting an unreliable automated score as the final result.**

## Demo

The project demonstration covers the five behaviors required for the assignment:
```
Knowledge-base question with citations
Order lookup
Multi-turn conversation
Correct refusal / human-support recommendation
Evaluation suite execution
```

https://github.com/user-attachments/assets/9ac583c9-7e6e-4a57-bc57-4e1386d6731f

## Coding Tools Used
ChatGPT

ChatGPT was used during development for:
```
RAG architecture planning
Python implementation assistance
Debugging the agent
Designing and improving the evaluation suite
Developing regression tests
Improving order-ID extraction
Investigating evaluation failures
Git/GitHub setup assistance
README and documentation preparation
Example of an Incorrect AI-generated Suggestion
```
An early evaluation implementation relied too heavily on exact phrase matching.

For example, the evaluator expected:

45 calendar days

but the agent produced:

45-calendar-day return window

Although the response communicated the intended policy correctly, the evaluator marked it as a failure.

This was an example of an AI-generated implementation being too rigid for natural-language evaluation.

The evaluation logic was subsequently improved to account for valid linguistic variations while maintaining deterministic checks for important requirements.

This highlighted an important lesson:

AI-generated code should be tested against real system behavior rather than accepted without verification.

## Conclusion

The Aster & Row Support Agent demonstrates a grounded customer-support architecture combining:
```
-Large language models
-Retrieval-Augmented Generation
-Vector search
-Tool calling
-Conversation memory
-Source citations
-Safety and privacy controls
-Human escalation
-Automated and manual evaluation
```
The final implementation was manually validated against all 15 visible evaluation cases, with 15/15 cases passing.
The system is designed not only to answer questions, but also to recognize when the available information is insufficient or conflicting and avoid producing an unjustifiably confident answer.







