"""Few-shot examples.

The tutorial this structure came from teaches few-shot with happy/sad emotion
pairs. The technique is right; the content is not. What we actually need the
model to learn is the difference between a question a senior engineer asks and
a question a quiz asks — that distinction is the whole brief ("should resemble
a real technical interview rather than a scripted questionnaire").

Each example is deliberately contrastive: the BAD form is named so the model
learns the boundary, not just the target.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# ---------------------------------------------------------------------------
# Question authoring
# ---------------------------------------------------------------------------

QUESTION_EXAMPLES: list[dict[str, str]] = [
    {
        "brief": (
            "kind=probe | day 8 Vector Databases Overview | tools: ChromaDB, Pinecone | "
            "candidate passed on attempt 2 | belief 0.55"
        ),
        "question": (
            "You had both Chroma and Pinecone running for the same knowledge base. "
            "When you picked one for the chatbot, what actually decided it?"
        ),
    },
    {
        "brief": (
            "kind=verification | day 28 Docker & Kubernetes Deployment | "
            "candidate is a DevOps Engineer with 10 years | passed first attempt | belief 0.82"
        ),
        "question": (
            "You deployed the chatbot to Kubernetes. What did you set the readiness probe to "
            "actually check, and what broke the first time you got it wrong?"
        ),
    },
    {
        "brief": (
            "kind=follow-up | previous answer mentioned chunk overlap fixing bad retrieval | "
            "day 6 Building the Knowledge Base | belief 0.68"
        ),
        "question": (
            "You said overlap fixed it. How did you land on that size rather than a larger one, "
            "and what did the larger one cost you?"
        ),
    },
    {
        "brief": (
            "kind=scenario | day 10 Retrieval & Matching Engine | "
            "candidate passed after 4 attempts | belief 0.45"
        ),
        "question": (
            "Your router is sending 'how much did I pay in claims last year' to vector search "
            "instead of SQL, and the answers sound confident but wrong. Where do you look first?"
        ),
    },
    {
        "brief": (
            "kind=misconception | day 11 RAG End-to-End | belief 0.5 | "
            "state a subtly wrong premise and see whether they correct it"
        ),
        "question": (
            "Since the chatbot only answers from retrieved context, grounding it in the vector "
            "store basically removes the hallucination problem, right?"
        ),
    },
    {
        "brief": (
            "kind=recovery | candidate has struggled on the last two questions | "
            "day 7 Embeddings Explained | belief 0.2"
        ),
        "question": (
            "Let's step back a level. In your own words, what does an embedding actually give "
            "you that a keyword search does not?"
        ),
    },
    {
        "brief": (
            "kind=opening | day 16 Chatbot Backend & API Integration | passed first attempt | "
            "belief 0.85 | give them a win"
        ),
        "question": (
            "Let's start somewhere you spent real time. Walk me through what happens inside your "
            "/chat endpoint from the moment a request lands."
        ),
    },
]

_question_example_prompt = ChatPromptTemplate.from_messages(
    [("human", "{brief}"), ("ai", "{question}")]
)

QUESTION_FEWSHOT = FewShotChatMessagePromptTemplate(
    example_prompt=_question_example_prompt,
    examples=QUESTION_EXAMPLES,
)


# ---------------------------------------------------------------------------
# What NOT to produce — stated once, in the system prompt, rather than as
# examples. Showing bad questions as assistant turns teaches the model to
# produce them.
# ---------------------------------------------------------------------------

ANTI_PATTERNS = """\
Never ask any of these:
- "What is X?" or "Can you explain X?" — that tests vocabulary, not understanding.
- "Have you used X?" — answerable with yes.
- Two questions at once. Ask one thing.
- A question that restates the day's title back at them.
- Anything that begins "As a <job title>, ..." — they know their own job.
- Anything longer than about thirty words. It has to be answerable out loud."""


# ---------------------------------------------------------------------------
# Scoring calibration
# ---------------------------------------------------------------------------

SCORING_ANCHORS = """\
Calibration anchors, so scores mean the same thing every turn:

correctness 0.9 — names the real mechanism and it is right
correctness 0.5 — the shape is right, the detail is absent or hazy
correctness 0.1 — confidently wrong, or answers a different question

depth 0.9 — explains WHY it behaves that way, names a trade-off they accepted
depth 0.5 — describes WHAT it does without the causal step
depth 0.1 — vocabulary only

specificity 0.9 — their own build: numbers, parameters, something that broke
specificity 0.5 — concrete but generic, could be from any tutorial
specificity 0.1 — entirely abstract

Two cases that must not be scored the same:
- An honest "I don't know, I never implemented that" is low correctness but is
  NOT dishonest. Say so in the note and keep the signal mild (about -0.3).
- Fluent, confident prose that names ownership and scale ("I built the whole
  pipeline end to end, in production, at scale") while containing no mechanism,
  no number and no trade-off is a bluff. Score it BELOW the honest admission."""
