# METRON — User Guide

---

## 1. What METRON Does For You

You have built an AI application — a chatbot, a RAG assistant, a multi-agent system. Before you let real users interact with it, you need to know: will it answer correctly? Can someone trick it into leaking data? Does it slow down under load? Is it biased?

METRON answers all of these by acting like a testing team that never sleeps. You give it your AI endpoint, describe what your app does, and METRON generates realistic user personas, runs hundreds of conversations, and measures the results across functional quality, security vulnerabilities, response quality, performance, and load.

At the end you get a health score, a breakdown by test category, and a list of exactly what failed and why.

---

## 2. Before You Start — What You Need

Before opening METRON, have these ready:

**Your AI endpoint**
This is the URL METRON will send test messages to. It must be a live, running HTTP endpoint that accepts POST requests. For example: `https://my-bot.company.com/chat`

**Your request and response field names**
METRON needs to know how your endpoint expects the message:
- Request field: what key to put the user message under. Default is `message`. If your endpoint expects `{"input": "hello"}`, the request field is `input`.
- Response field: where to find the AI reply in the response. Default is `response`. If your endpoint returns `{"data": {"text": "Hello!"}}`, the response field is `data.text` (dot notation works).

**An API key for your endpoint** (if it uses bearer token auth)

**A description document about your AI application** — a plain text file (.txt) describing what your app does, who uses it, what it can and cannot do. The richer this document, the better the test personas and prompts METRON generates. See Section 3 for what to include.

**An LLM API key for METRON itself**
METRON uses its own LLM to generate test personas and evaluate results. You need one of:
- NVIDIA NIM API key (free, recommended for first run)
- Groq API key (free, 100K tokens/day)
- Google Gemini API key (free, 1M tokens/day)
- Azure OpenAI API key + endpoint (if you have Azure access — highest quality)

**For RAG applications additionally:**
A ground truth file (CSV or JSON) with the questions your RAG should be able to answer, the expected answers, and the relevant context chunks. See Section 8 for the exact format.

---

## 3. How to Write a Good Description Document

The description document is the most important input. METRON uses it to understand your app and generate realistic test users. A vague document produces generic tests. A specific document produces tests that closely match real user behavior.

**What to include:**

**What the app does** — describe it in plain language as if explaining to a new employee.
> "This is a healthcare chatbot for patients of City Medical Group. It helps patients find nearby clinic locations, check appointment availability, understand common medication side effects, and get general wellness advice."

**Who uses it** — list all the different types of people who interact with it.
> "Users include: adult patients (most common), elderly patients who are less tech-savvy, caregivers managing appointments for family members, and medical staff checking on patient queries."

**What it cannot do** — this is critical. METRON will test these boundaries.
> "The app cannot prescribe medications, provide specific diagnoses, access patient records, or schedule appointments directly. It redirects these requests to the clinic's scheduling line."

**Domain-specific vocabulary** — words your users would actually use.
> "Patients often say: 'copay', 'referral', 'in-network', 'prior authorization', 'primary care physician'. They avoid medical jargon but may use common terms like 'blood pressure check' or 'flu shot'."

**What a good response looks like**
> "A good response provides a direct answer, is under 150 words, does not use medical abbreviations, and offers a next step (e.g., 'call this number', 'visit this page')."

**Format recommendation:**
Plain text (.txt), 300–2000 words. Bullet points are fine. No PDF, no Word documents — paste or export to plain text.

---

## 4. Step-by-Step: Running Your First Test

### Step 1 — Create a Project

Open METRON at `http://localhost:3000`.

Click **Connect Project**. Fill in:
- **Project name** — anything descriptive ("Healthcare Bot v2 Test")
- **Endpoint URL** — your AI's live URL
- **API key** — your endpoint's bearer token (leave blank if no auth)
- **Upload document** — drag your description .txt file here

Click **Connect**. Your project appears in the grid.

### Step 2 — Configure the Test

Click your project, then click **Configure**.

Work through the sections:

**Endpoint Configuration**
- Verify the endpoint URL is correct
- Set the **Request field** (default: `message`)
- Set the **Response field** (default: `response`) — use dot notation for nested fields
- Set auth type (None or Bearer Token) and paste your token if needed

Click **Test Connection** — METRON will send a test ping to verify the endpoint is alive and responding. You will see either "Connected successfully" or an error message telling you what went wrong.

**Agent Configuration**
- **Agent name** — a short label for reports (e.g., "Healthcare Bot")
- **Domain** — type your domain (healthcare, finance, legal, retail, etc.). This affects how METRON weights security vs performance in your health score.
- **Description** — a one-paragraph summary of what your app does
- **Application type** — select Chatbot or RAG

**RAG Configuration** (only visible if you selected RAG)
- Toggle "This is a RAG Agent" on
- Upload your ground truth file (CSV or JSON with question, expected_answer, context columns — see Section 8)

**Test Parameters**
- **Personas** — how many synthetic users to generate (default 3, max 50). More personas = more thorough coverage but longer test time.
- **Scenarios** — test prompts per persona run (default 5)
- **Conversation Turns** — how many back-and-forth exchanges per test (default 3, max 15). Longer conversations test how the AI handles follow-ups and edge cases.
- **Performance Requests** — how many requests to send for latency measurement (default 20)
- **Load Test Users** — concurrent users for load test (default 5)
- **Load Duration** — how long to run the load test in seconds (default 30)

**LLM Provider** (for METRON's own reasoning)
- Select your provider
- Paste your API key
- The description next to each provider shows the RPM limit and cost model

**Security Settings**
Select which attack categories to test. All are selected by default:
- **Jailbreak** — attempts to make the AI ignore its instructions
- **Prompt Injection** — tries to hijack the AI's behavior via user input
- **PII Extraction** — tries to get the AI to reveal personal information
- **Toxicity** — attempts to make the AI produce harmful content
- **Social Engineering** — manipulative but plausible-sounding requests
- **Encoding** — attacks using character substitution or encoding tricks

**Attacks Per Category** — how many attack variants per category (default 3, max 20)

**Quality Settings**
- RAGAS metrics: faithfulness, answer_relevancy (both selected by default)
- DeepEval metrics: hallucination, toxicity
- GEval: domain-specific quality scoring (enabled by default)

**Feedback Loop** — if enabled, METRON runs a second pass targeting the specific areas where failures were found in the first pass

Click **Next** when ready.

### Step 3 — Review the Test Plan

The Preview page shows exactly what METRON is about to run:

**Tool Status** — shows which evaluation tools are installed (green = installed, red = missing). Missing tools mean those specific metrics will be skipped. The core pipeline runs regardless.

**Test Counts** — how many functional, security, quality, performance, and load tests will run.

**Generated Personas** — the synthetic users METRON will use. Each persona card shows:
- Name and background
- Expertise level (novice / intermediate / expert)
- Emotional state (calm / frustrated / urgent)
- Intent (genuine / adversarial / edge case)
- Sample messages in their voice

Take a moment to read these. If a persona doesn't make sense for your application, you can go back and update the description document.

**Test Scenarios** — sample prompts from each persona. These are the actual messages that will be sent to your AI.

When you are satisfied, click **Run Tests**.

### Step 4 — Watch the Live Feed

The Run page shows real-time progress as METRON works. You will see:

**Phase headers** — as each stage starts (Persona Generation, Test Generation, Running Conversations, Evaluating Results, etc.)

**Persona cards** — each generated persona appears as it's created, showing their name, goal, and behavioral traits

**Test prompts** — functional and security prompts as they are generated, showing which persona they belong to and what the expected behavior should be

**Conversation feed** — as conversations run, you see:
- The message sent to your AI
- Your AI's response
- Which persona sent it
- Latency in milliseconds
- How many turns the conversation took

**Evaluation results** — as each batch of evaluations completes, a summary card shows pass/fail counts and average score

The progress bar at the top shows overall completion percentage. A typical run takes 3–6 minutes depending on your provider and how many personas you configured.

If the run fails, an error card appears with a specific message explaining what went wrong.

### Step 5 — Read the Results

When the run completes, METRON automatically navigates to the Results page.

---

## 5. Understanding the Results Page

### The Health Score

The large gauge at the top is your overall health score (0–100).

```
90–100   Excellent — production ready
70–89    Good — minor issues to address
50–69    Fair — significant issues found
0–49     Poor — not ready for production
```

The health score is **domain-weighted**. For a healthcare app, security failures pull the score down more than performance issues. For a retail app, performance and load matter more. The domain you specified determines the weighting.

**Passed / Failed badge** — green "PASSED" means health score ≥ 70 and you cleared the threshold. Red "FAILED" means the score was below 70 or a critical failure was found.

### The Test Class Tabs

Results are organized into tabs: Functional, Security, Quality, RAG (if applicable), Performance, Load.

Each tab shows:
- **Pass rate** — percentage of tests that passed
- **Average score** — mean score across all tests (0–100)
- **By metric** — breakdown of each metric separately

Click any row in the results table to see:
- The full message sent to your AI
- Your AI's full response
- The score and whether it passed
- The reasoning from the evaluation tool

### Functional Tab

Shows how well your AI answers the questions it's supposed to answer.

**Hallucination** — did your AI make up information not supported by its knowledge? A score close to 100 means low hallucination.

**Answer Relevancy** — did the response actually address what was asked? A low relevancy score means the AI is going off-topic.

**Usefulness** — is the response actionable? Does it help the user complete their goal?

**LLM Judge** — domain-specific scoring against the criteria extracted from your document. For healthcare, this might include "clinical accuracy", "empathy", "clarity". Each criterion is scored separately.

### Security Tab

Shows how your AI handles attacks and adversarial inputs.

**PII Leakage** — did your AI reveal personal identifiable information (SSN, email, phone, credit card)? Detected using Presidio, a dedicated PII scanner. Score of 100 = no PII found in any response.

**Toxicity (Output)** — did your AI produce toxic, insulting, or threatening content in any response? Scored using Detoxify, a BERT classifier. Score of 100 = no toxic output.

**Prompt Injection** — were injected instructions detected in attack prompts? Scored using LLM Guard (DeBERTa model). Score of 100 = attack detected and resisted.

**Bias & Fairness** — does your AI treat different user groups differently? Scored using DeepEval's BiasMetric.

**Toxic Request** — when sent a known-harmful request (from the HarmBench dataset), did your AI refuse? Score 100 = always refused. Score 0 = complied with harmful requests.

**Attack Resistance** — evaluated by an LLM judge: did your AI maintain its safety guidelines when attacked? Considers whether it refused cleanly, whether it was compromised, and whether the response quality remained professional.

**What to look for:**
- Any PII leakage score below 100 is a critical finding — your AI is revealing sensitive data.
- Attack resistance below 70 means the AI can be manipulated.
- Toxic request below 80 means the AI sometimes complies with harmful requests.

### Quality Tab

Shows the quality of your AI's responses on a deeper level.

**GEval scores** — domain-specific quality criteria scored individually. If your document mentioned "responses should be under 150 words and provide a next step", GEval tests exactly this.

**RAGAS metrics** (RAG mode only, also appears in RAG tab):
- **Faithfulness** — is the response supported by the retrieved context? A low score means hallucination relative to the knowledge base.
- **Context Recall** — does the retrieved context contain the answer to the question?
- **Context Precision** — is the context that was retrieved actually relevant to the question?

### RAG Tab (RAG mode only)

Appears when you ran in RAG mode with a ground truth file. Shows all 5 RAG metrics:
- Faithfulness (RAGAS) — responses grounded in context
- Context Recall (RAGAS) — context contains the answer
- Context Precision (RAGAS) — context is relevant
- Answer Relevancy (DeepEval) — answer addresses the question
- Context Relevancy (DeepEval) — context is on-topic

Each row shows the specific question, the AI's answer, and why it passed or failed. This is the most useful tab for understanding your RAG system's accuracy.

### Performance Tab

Shows how fast your AI responds under normal conditions.

- **Avg / P95 / P99 Latency** — average, 95th percentile, and 99th percentile response times in milliseconds. P95 means 95% of requests were faster than this number.
- **Throughput** — requests per second your AI can handle
- **Error rate** — what percentage of requests failed

**What's acceptable:** For most apps, p95 latency under 3000ms is good. Above 5000ms is concerning. An error rate above 5% means the endpoint is unreliable.

### Load Tab

Shows how your AI behaves when many users hit it simultaneously.

- **Concurrent users** — how many simultaneous users were simulated (what you configured)
- **Total requests** — total sent during the test
- **Error rate** — requests that failed under load
- **P95 latency** — 95th percentile under load (this will be higher than the performance tab)
- **Requests per second** — actual throughput achieved

**What to look for:** Compare load P95 latency to performance P95 latency. If load latency is more than 3× the performance latency, your AI doesn't scale well. An error rate above 10% under load means the endpoint starts dropping requests at this concurrency level.

### Persona Breakdown

At the bottom of the results page, each persona is listed with their overall pass rate. This tells you **which type of user** your AI struggles with.

If novice frustrated users have a 40% pass rate but expert calm users have 95%, your AI is not handling confused or emotional users well.

Adversarial personas will always have lower pass rates — that's expected. The question is whether they're below 50%, which would indicate actual vulnerabilities.

### Failure Drill-Down

The top 20 worst-performing tests are listed in detail. Each shows:
- Which metric failed and by how much
- The exact message that was sent
- The AI's full response
- The reason for failure (from the evaluation tool)
- Which persona was involved

This is your action list. Start from the top and work down.

---

## 6. What to Do With the Results

**If hallucination is high:**
Your AI is making up information. Check whether your AI has access to the right knowledge base. If it's a RAG system, check whether the retrieval is working correctly — are the right chunks being retrieved?

**If answer relevancy is low:**
Your AI is going off-topic. Review the conversations where this happened (click the row to see the full exchange). Is the AI being redirected by something in the prompt? Does it misunderstand the question?

**If PII leakage is detected:**
This is critical. Look at exactly what was leaked (the failure drill-down shows the full response). Check whether your AI has access to any stored user data. Consider adding a PII filtering layer on the output.

**If attack resistance is low:**
Look at which attack categories succeeded. If jailbreak attacks are getting through, your system prompt needs stronger boundary reinforcement. If prompt injection is working, your app is processing user input as instructions — this needs an architectural fix.

**If performance p95 is above 5000ms:**
Check your AI endpoint's infrastructure. Is it using a GPU? Is the model too large for the hardware? Consider caching common responses or using a faster model tier.

**If load test error rate is above 10%:**
Your endpoint doesn't scale to the configured concurrent user count. This might mean: queue limits are too low, the endpoint needs horizontal scaling, or connection pooling is not configured.

**If RAG faithfulness is low:**
Your AI is generating responses not supported by the retrieved context. This is hallucination in RAG mode. Possible causes: the retriever is returning irrelevant chunks, the prompt template doesn't instruct the model to stay grounded in context, or the model is trained to be overly helpful and fills gaps with invented information.

**If context recall is low:**
Your retriever is not finding the right chunks to answer the question. Possible causes: embedding model mismatch, chunk size is too small or too large, the ground truth context you provided doesn't match how your app actually retrieves.

---

## 7. Running Tests Again After Fixes

Each time you fix something, run another test from the same project. The configure page remembers your last settings. You can adjust specific parameters without re-entering everything.

Track your health score across runs. A genuine improvement will show in the score. Watch particularly for regression — fixes in one area sometimes introduce failures in another (e.g., making responses shorter for latency improvements might reduce answer completeness).

---

## 8. Ground Truth File Format (RAG Mode)

The ground truth file is required for full RAG evaluation. It contains the questions your RAG system should be able to answer, the correct answers, and the relevant context.

**JSON format:**
```json
[
  {
    "question": "What are the side effects of ibuprofen?",
    "expected_answer": "Common side effects include stomach upset, heartburn, nausea, and dizziness. Serious side effects include kidney problems and increased risk of heart attack.",
    "context": "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID). Common side effects include gastrointestinal issues such as stomach upset, heartburn, and nausea. Dizziness and headache may also occur. Serious side effects include renal impairment with prolonged use and cardiovascular risk including myocardial infarction."
  },
  {
    "question": "What is the maximum daily dose of acetaminophen for adults?",
    "expected_answer": "The maximum daily dose of acetaminophen for adults is 4000mg (4g) per day, though 3000mg is recommended for regular use.",
    "context": "Acetaminophen dosage guidelines: Adults and children over 12 years: 325-650mg every 4-6 hours as needed. Maximum daily dose: 4000mg (4 grams). For chronic use or patients with liver concerns, maximum 3000mg per day is recommended. Do not exceed the maximum daily dose."
  }
]
```

**CSV format:**
```csv
question,expected_answer,context
What are the side effects of ibuprofen?,Common side effects include stomach upset...,"Ibuprofen is a nonsteroidal anti-inflammatory drug..."
What is the maximum daily dose of acetaminophen?,The maximum daily dose is 4000mg...,"Acetaminophen dosage guidelines: Adults..."
```

**Tips for writing good ground truth:**
- Use questions that represent what real users actually ask
- Expected answers should be complete and accurate — METRON scores your AI against these
- Context should be the actual text from your knowledge base that contains the answer — copy it directly from your source documents
- Include at least 10–20 pairs for meaningful evaluation
- Include questions at different difficulty levels: easy (direct lookup), medium (requires synthesis), hard (requires combining multiple sources)
- Include questions the system should refuse or say "I don't know" to — this tests boundary behavior
- The context field is what METRON uses for faithfulness scoring — if context is missing, faithfulness cannot be measured

---

## 9. Common Problems and Solutions

**"Test connection failed"**
- Check the endpoint URL is correct and the service is running
- Verify the request field name matches what your API expects
- Check whether your API requires authentication — add the bearer token
- Make sure the endpoint is accessible from the machine running METRON (no firewall blocks)

**"Pipeline failed" error on the run page**
- Most common cause: the LLM API key is wrong or has expired. Double-check on the configure page.
- Second cause: the endpoint URL stopped responding mid-run. Check if your AI service is still running.
- If the error message says "rate limit" or "429": you are hitting your LLM provider's limit. Either wait and retry, or switch to a provider with higher RPM.

**Results show all zeros or most tests skipped**
- Check the Tools Status on the Preview page. If DeepEval is not installed, functional and quality metrics will be empty. If Presidio is not installed, PII leakage will not run.
- To install missing tools: `pip install deepeval presidio-analyzer presidio-anonymizer detoxify llm-guard`
- For Presidio to work, you also need: `python -m spacy download en_core_web_lg`

**RAG tab not appearing in results**
- Make sure you toggled "This is a RAG Agent" on in the Configure page
- Make sure you uploaded a ground truth file before clicking Next
- Check the run page — look for a "RAG Evaluation" phase header. If it appears but shows 0 metrics, the conversations did not have retrieved context — your endpoint may not be returning context in the expected format.

**Personas don't make sense for my app**
- This means the description document needs more detail about who uses the app. Add specific user roles, what they know, what they don't know, and what they're trying to accomplish.
- Re-upload the document, go back through Configure, and run again.

**Load test shows timeout or 147s buffer error**
- The load duration combined with user count requires more time than allocated. The fix is already in the codebase (timeout scales with user count). If you see this, check that you're running the latest version.
- For very high user counts (50+), increase the Load Duration slider.

**Security tests feel too generic**
- The attacks are adapted to your domain via LLM, but they need your document's domain vocabulary and use cases to be specific. A richer document produces more targeted attacks.
- Increasing "Attacks per Category" gives more variety.

---

## 10. What Good Results Look Like in Practice

Here is a realistic example of what you might see after a first run on a customer support chatbot:

```
Health Score: 71   [PASSED]

Functional:   78% pass rate, avg score 0.74
  Hallucination:      89% pass rate   ← good
  Answer Relevancy:   72% pass rate   ← acceptable
  Usefulness:         65% pass rate   ← needs work
  LLM clarity:        81% pass rate   ← good

Security:     62% pass rate, avg score 0.61
  PII Leakage:        100% pass rate  ← excellent
  Toxicity:           98% pass rate   ← excellent
  Attack Resistance:  58% pass rate   ← needs work
  Toxic Request:      70% pass rate   ← acceptable
  Prompt Injection:   55% pass rate   ← needs work

Quality:      69% pass rate, avg score 0.71
  GEval tone:         85% pass rate
  GEval completeness: 55% pass rate   ← needs work

Performance:  p95 = 1850ms, error rate 1.2%  ← good
Load (20 users): p95 = 4200ms, error rate 3.1%  ← acceptable
```

**What to fix first:**
1. Attack Resistance (58%) — the most critical security gap. Look at which attacks succeeded.
2. Prompt Injection (55%) — another security concern. Check the failure drill-down.
3. Completeness (55%) — responses are missing information. Look at which questions this happened on.
4. Usefulness (65%) — responses are technically correct but not actionable. Often means the AI is giving vague answers instead of specific steps.

After addressing these, a second run typically shows:
- Attack Resistance rises to 70–80% after strengthening the system prompt
- Completeness rises after adding more content to the knowledge base
- Health score rises from 71 to 78–82

This iterative process — run, review, fix, run again — is how METRON is meant to be used. It is not a one-time check but a part of your development and deployment workflow.
