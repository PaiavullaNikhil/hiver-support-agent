# AI Support Agent for AppleSupport

A retrieval-grounded AI customer-support agent built from the Kaggle **Customer Support on Twitter** dataset.

The system is designed to assist AppleSupport-style customer interactions by combining:

- intent classification,
- historical support-case retrieval,
- grounded response generation,
- deterministic escalation,
- automated evaluation,
- LLM-as-judge evaluation,
- and human-vs-judge agreement analysis.

The project intentionally focuses on the support-agent decision loop rather than building a production customer-support platform.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Project Goal](#2-project-goal)
3. [Solution Overview](#3-solution-overview)
4. [System Architecture](#4-system-architecture)
5. [Dataset](#5-dataset)
6. [Brand Selection](#6-brand-selection)
7. [Data Processing](#7-data-processing)
8. [Conversation Construction](#8-conversation-construction)
9. [Intent Taxonomy](#9-intent-taxonomy)
10. [Golden Evaluation Set](#10-golden-evaluation-set)
11. [Intent Classification](#11-intent-classification)
12. [Classifier Experiments](#12-classifier-experiments)
13. [Final Intent Classifier](#13-final-intent-classifier)
14. [Historical Retrieval](#14-historical-retrieval)
15. [Response Generation](#15-response-generation)
16. [Escalation Policy](#16-escalation-policy)
17. [End-to-End Agent](#17-end-to-end-agent)
18. [Evaluation Framework](#18-evaluation-framework)
19. [LLM-as-Judge](#19-llm-as-judge)
20. [Human vs LLM-Judge Agreement](#20-human-vs-llm-judge-agreement)
21. [Failure Analysis](#21-failure-analysis)
22. [What Is Misleading About the Headline Number](#22-what-is-misleading-about-the-headline-number)
23. [Limitations](#23-limitations)
24. [Decision Log](#24-decision-log)
25. [Project Structure](#25-project-structure)
26. [Environment Setup](#26-environment-setup)
27. [Dataset Setup](#27-dataset-setup)
28. [Reproducing the Evaluation](#28-reproducing-the-evaluation)
29. [Running Failure Analysis](#29-running-failure-analysis)
30. [Running the Agent](#30-running-the-agent)
31. [Running Tests](#31-running-tests)
32. [Generated Artifacts](#32-generated-artifacts)
33. [What Is Not Built](#33-what-is-not-built)
34. [Next Week](#34-next-week)
35. [Summary](#35-summary)

---

# 1. Problem Statement

Customer-support conversations on social media are difficult to automate reliably because they are often:

- short,
- noisy,
- informal,
- misspelled,
- context-dependent,
- multi-issue,
- and mixed with irrelevant social-media language.

A useful support agent should do more than generate a plausible response.

It should:

1. identify the customer's primary support issue,
2. find evidence from how the brand historically handled similar issues,
3. generate a response grounded in that evidence,
4. avoid unsupported claims,
5. recognize uncertainty,
6. and escalate cases that should not be automatically handled.

For this project, the target support brand is **AppleSupport**.

---

# 2. Project Goal

The goal is to build a support-agent prototype that performs the following workflow:

```text
Customer Message
       |
       v
Intent Classification
       |
       v
Historical Support Retrieval
       |
       v
Grounded Response Generation
       |
       v
Escalation Decision
       |
       +-----------> Auto-handle
       |
       +-----------> Human escalation
```

The system therefore separates:

- understanding the issue,
- finding historical evidence,
- generating the response,
- and deciding whether automation is appropriate.

This separation makes the system easier to inspect and evaluate.

---

# 3. Solution Overview

The final system contains four major components.

## 3.1 Intent Classifier

A supervised classifier predicts one of 12 support intents.

The final model uses:

```text
Word TF-IDF
      +
Character TF-IDF
      |
      v
Linear SVM
```

Previous-message context is included when available.

---

## 3.2 Historical Retrieval

Historical AppleSupport interactions are embedded using:

```text
all-MiniLM-L6-v2
```

The embeddings are stored in a FAISS index.

For a new customer request, the system retrieves the top-5 most similar historical support cases.

---

## 3.3 Response Generation

Gemini receives:

- the customer's message,
- previous-message context when available,
- predicted intent,
- retrieved historical cases,
- and historical AppleSupport responses.

The model generates a concise support response grounded in the retrieved evidence.

---

## 3.4 Escalation

Escalation is determined using deterministic rules rather than allowing the LLM to independently decide whether to escalate.

Signals include:

- classifier confidence,
- historical retrieval similarity,
- vague requests,
- unclear intent,
- account/authentication issues,
- device-specific issues.

---

# 4. System Architecture

```text
                         Customer Message
                                |
                                v
                 +-----------------------------+
                 |      Intent Classifier      |
                 |                             |
                 | Word TF-IDF                 |
                 | Character TF-IDF            |
                 | Linear SVM                  |
                 +-------------+---------------+
                               |
                               v
                       Predicted Intent
                               |
                +--------------+--------------+
                |                             |
                v                             v
      +---------------------+       +---------------------+
      | Historical          |       | Escalation Policy   |
      | Retrieval           |       |                     |
      |                     |       | Classifier margin   |
      | MiniLM embeddings   |       | Similarity          |
      | FAISS               |       | Ambiguity           |
      +----------+----------+       | Account/device      |
                 |                  +---------------------+
                 v
        Top Historical Cases
                 |
                 v
      +-----------------------+
      | Gemini Response       |
      | Generator             |
      +-----------+-----------+
                  |
                  v
             Draft Reply
                  |
                  v
          Final Agent Decision
                  |
          +-------+-------+
          |               |
          v               v
     Auto-handle      Escalate
```

---

# 5. Dataset

The project uses the Kaggle **Customer Support on Twitter** dataset.

Dataset:

```text
thoughtvector/customer-support-on-twitter
```

The dataset contains Twitter-based customer-support conversations involving customers and brand support accounts.

The main columns are:

| Column | Description |
|---|---|
| `tweet_id` | Tweet identifier |
| `author_id` | Tweet author/support account |
| `inbound` | Whether the tweet is inbound to the brand |
| `created_at` | Tweet timestamp |
| `text` | Tweet content |
| `response_tweet_id` | Response tweet relationship |
| `in_response_to_tweet_id` | Parent tweet relationship |

The full dataset contains approximately **2.8 million tweets**.

The project does not train on the complete dataset. Instead, it extracts the required AppleSupport interactions.

---

# 6. Brand Selection

The dataset contains many different support accounts.

An initial scan identified more than 100 candidate support accounts.

Several high-volume candidates were compared, including:

```text
AmazonHelp
AppleSupport
Uber_Support
SpotifyCares
Delta
```

AppleSupport was selected because it provides:

- a large historical support corpus,
- technical troubleshooting requests,
- software-update issues,
- connectivity problems,
- account issues,
- device issues,
- Apple-service issues,
- and useful historical support responses.

This creates a useful environment for both intent classification and historical-response retrieval.

---

# 7. Data Processing

The raw dataset contains individual tweets rather than ready-to-use customer/support examples.

The project therefore constructs customer-to-AppleSupport response pairs.

The cleaned dataset contains:

```text
106,523 interactions
```

The processed representation contains:

```text
customer_tweet_id
customer_message
previous_message
has_context
parent_author
brand_response
```

---

## 7.1 Customer/Brand Pair Construction

For each AppleSupport response, the system identifies the customer tweet being answered using:

```text
in_response_to_tweet_id
```

The corresponding customer tweet text is then joined to the AppleSupport response.

This produces:

```text
Customer message
       +
AppleSupport response
```

which forms the core historical support case.

---

## 7.2 Previous-Message Context

Some customer tweets are themselves replies to another tweet.

When that information is available, the preceding message is retained.

Approximately 30% of the cleaned AppleSupport interactions contain usable preceding-message context.

This is important because support conversations frequently contain messages such as:

```text
same issue
```

which cannot be understood reliably without conversational context.

---

## 7.3 Cleaning

The preprocessing step removes:

- empty customer messages,
- empty brand responses,
- obvious low-information messages,
- duplicate customer/response combinations.

Short but meaningful messages are retained.

---

# 8. Conversation Construction

The resulting interaction structure is:

```text
Previous message
       |
       v
Customer message
       |
       v
AppleSupport response
```

The previous message is optional.

For classification:

```text
Previous message + Customer message
```

is used when context is available.

The historical AppleSupport response is deliberately excluded from the classifier input.

---

## Leakage Prevention

The response text can reveal the issue being discussed.

Therefore:

```text
Customer message + previous context
             |
             v
       Intent classifier
```

is used instead of:

```text
Customer message
+
AppleSupport response
             |
             v
       Intent classifier
```

This prevents target leakage.

---

# 9. Intent Taxonomy

A compact taxonomy of 12 support intents was created.

| Intent | Definition |
|---|---|
| `ios_update` | iOS update failures, update-related problems, or iOS-version behavior |
| `battery_charging` | Battery drain, charging, power consumption, or battery health |
| `connectivity` | Wi-Fi, cellular, Bluetooth, network, or connection problems |
| `account_authentication` | Apple ID, login, password, authentication, or account access |
| `icloud_backup` | iCloud storage, backup, syncing, or recovery |
| `apps_app_store` | Apps, App Store, installation, updates, crashes, or app behavior |
| `audio` | Speakers, microphone, calls audio, headphones, AirPods, or sound |
| `camera_photos` | Camera, photos, video, or photo-library issues |
| `display_input` | Screen, touchscreen, keyboard, typing, or UI interaction |
| `apple_services` | Apple services such as iTunes, Apple Music, iMessage, etc. |
| `device_hardware` | Device-level issues not better covered elsewhere |
| `other_unclear` | Vague, incomplete, unrelated, or insufficiently specified requests |

---

## Primary Intent Policy

The labeling policy is:

> Label the issue that is the primary reason for the customer's support request. If no issue clearly dominates, use `other_unclear`.

This is particularly important for multi-issue requests.

For example, a message may mention an iOS update while the actual problem is battery drain.

In that case, `battery_charging` is preferred if battery behavior is the primary support issue.

---

## Taxonomy Exploration

Keyword-based rules were initially explored to understand the dataset.

However, keyword matching showed substantial overlap.

In an initial sample:

```text
43 examples -> exactly one candidate intent
42 examples -> multiple candidate intents
15 examples -> no candidate intent
```

Therefore, keyword rules were used for exploration but not as the final classifier.

---

# 10. Golden Evaluation Set

A golden evaluation set of:

```text
200 examples
```

was created.

The examples were sampled from the cleaned AppleSupport dataset and then manually reviewed and labeled using the fixed taxonomy.

The set contains all 12 intents.

File:

```text
data/evaluation/apple_support_golden_200.csv
```

The golden set contains:

```text
customer_tweet_id
customer_message
previous_message
has_context
intent
notes
```

---

## Evaluation Isolation

The golden examples are excluded from:

- classifier training,
- historical retrieval.

This prevents direct evaluation leakage.

---

# 11. Intent Classification

Several conventional machine-learning approaches were evaluated before selecting the final classifier.

The evaluation uses stratified 5-fold cross-validation.

The primary metrics are:

- Accuracy
- Macro F1
- Weighted F1

Macro F1 is particularly important because the intent classes are unevenly represented.

---

# 12. Classifier Experiments

The following models were evaluated.

## Majority Baseline

The majority class provides a trivial reference point.

Accuracy:

```text
17.5%
```

---

## TF-IDF + Complement Naive Bayes

```text
Accuracy    : 41.5%
Macro F1    : 29.5%
Weighted F1 : 35.7%
```

---

## TF-IDF + Linear SVM

```text
Accuracy    : 45.0%
Macro F1    : 35.1%
Weighted F1 : 41.4%
```

---

## TF-IDF + Logistic Regression

```text
Accuracy    : 45.0%
Macro F1    : 35.2%
Weighted F1 : 42.2%
```

---

## Word + Character TF-IDF + Linear SVM

```text
Accuracy    : 49.5%
Macro F1    : 42.7%
Weighted F1 : 46.8%
```

This was selected as the final classifier.

---

# 13. Final Intent Classifier

The final model uses:

### Word features

```text
TF-IDF
1-2 word n-grams
```

### Character features

```text
Character TF-IDF
3-5 character n-grams
```

### Classifier

```text
LinearSVC
class_weight="balanced"
C=1.0
```

Character features are useful for social-media text because they can capture partial word patterns, spelling variations, and noisy text.

The classifier also uses the previous message when available.

---

## Final Classification Result

On the 200-example golden set:

```text
Accuracy    : 49.5%
Macro F1    : 42.7%
Weighted F1 : 46.8%
```

These are development evaluation results and should not be interpreted as production performance.

---

# 14. Historical Retrieval

The response-generation component retrieves historically similar AppleSupport cases.

The retrieval model is:

```text
all-MiniLM-L6-v2
```

Embeddings are normalized and stored in FAISS.

The search uses:

```text
Inner Product similarity
```

and retrieves:

```text
Top 5 historical cases
```

---

## Retrieval Corpus

The retrieval corpus contains approximately:

```text
106K historical cases
```

The 200 golden examples are excluded.

---

## Retrieval Index

The generated index is stored at:

```text
data/processed/retrieval_index/
```

The directory contains the persisted FAISS index and retrieval corpus.

The first build requires embedding the historical messages.

Later runs load the cached index.

This avoids repeatedly encoding more than 100,000 historical messages.

---

## Retrieval Diagnostic

A development diagnostic showed approximately:

```text
Mean top-1 similarity: 0.809
Median top-1 similarity: 0.811
Top-1 intent alignment: 82.0%
Top-5 intent alignment: 95.5%
```

Important:

These intent-alignment diagnostics were calculated using classifier predictions rather than independently human-labeled retrieval relevance.

They should therefore be treated as diagnostic evidence rather than ground-truth retrieval accuracy.

---

# 15. Response Generation

Gemini is used for response synthesis.

The model receives:

```text
Customer message
Previous message
Predicted intent
Historical customer cases
Historical AppleSupport responses
```

The generation prompt requires the model to:

- directly address the customer's issue,
- use historical evidence,
- avoid inventing policies,
- avoid unsupported guarantees,
- avoid unsupported troubleshooting claims,
- avoid exposing tweet IDs,
- avoid exposing social-media metadata,
- avoid blindly copying historical URLs.

The generator returns structured output:

```json
{
  "draft_reply": "...",
  "evidence_used": [...]
}
```

The purpose of the LLM is therefore primarily:

```text
Historical evidence
       +
Customer context
       |
       v
Grounded response synthesis
```

rather than unrestricted answer generation.

---

# 16. Escalation Policy

The system uses deterministic escalation rules.

The LLM is not solely responsible for deciding whether a case should be escalated.

The policy considers:

1. classifier decision margin,
2. retrieval similarity,
3. vague-message patterns,
4. unclear intent,
5. account/authentication issues,
6. device-specific issues.

---

## Classifier Confidence

The Linear SVM decision margin is calculated as:

```text
highest class score - second-highest class score
```

The development threshold is:

```text
0.30
```

Cases below this threshold can be escalated.

---

## Retrieval Similarity

The development retrieval threshold is:

```text
0.65
```

If no sufficiently similar historical case is found, the system can escalate.

---

## Vague Messages

Explicit vague patterns include examples such as:

```text
same issue
same problem
having the same
please help
help me
```

A message is not considered vague solely because it is short.

For example:

```text
My iPhone battery is draining very quickly.
```

contains enough information to identify a specific support issue.

---

## Account and Device Issues

The following intents receive more conservative handling:

```text
account_authentication
device_hardware
```

These cases may require information or actions that are not available to the prototype.

---

## Development Threshold Sweep

A development threshold sweep showed the following trade-off:

| Margin Threshold | Auto-handle Rate | Auto-handled Accuracy |
|---:|---:|---:|
| 0.01 | 96.0% | 51.6% |
| 0.02 | 90.0% | 53.3% |
| 0.05 | 80.5% | 57.1% |
| 0.10 | 64.0% | 64.1% |
| 0.15 | 55.5% | 67.6% |
| 0.20 | 49.5% | 70.7% |
| 0.30 | 35.5% | 84.5% |
| 0.40 | 26.5% | 86.8% |
| 0.50 | 20.5% | 90.2% |

The final development threshold was set to:

```text
0.30
```

This is a development trade-off and **not a production safety guarantee**.

---

# 17. End-to-End Agent

The complete agent performs:

```text
1. Validate customer message
2. Predict intent
3. Calculate classifier confidence
4. Retrieve historical cases
5. Calculate retrieval similarity
6. Generate grounded draft
7. Apply deterministic escalation policy
8. Return structured result
```

The final output contains:

```text
intent
classifier_margin
top_similarity
draft_reply
evidence_used
escalate
escalation_reason
retrieved_cases
```

---

# 18. Evaluation Framework

The project has multiple evaluation layers.

```text
                    Evaluation
                        |
        +---------------+---------------+
        |               |               |
        v               v               v
 Intent Metrics    Response Quality   Escalation
        |               |               |
        v               v               v
 Accuracy/F1       LLM Judge        Policy analysis
        |
        v
 Failure Analysis
```

This separates classifier performance from response-generation quality.

---

## Automated Intent Evaluation

Run:

```powershell
python -m src.run_evaluation
```

The command reports:

- baseline performance,
- final classifier performance,
- per-intent precision,
- per-intent recall,
- per-intent F1,
- top confusion pairs.

Results are saved to:

```text
data/evaluation/evaluation_results.json
```

---

# 19. LLM-as-Judge

A separate evaluation set of:

```text
30 generated responses
```

was evaluated using an LLM judge.

The judge does not receive the true intent label.

It evaluates the generated response using five dimensions.

| Dimension | Scale |
|---|---|
| Relevance | 1–5 |
| Grounding | 1–5 |
| Helpfulness | 1–5 |
| Overall quality | 1–5 |
| Unsupported claim | 0/1 |

---

## Judge Results

```text
Mean relevance        : 4.97 / 5
Mean grounding        : 5.00 / 5
Mean helpfulness      : 4.77 / 5
Mean overall quality  : 4.77 / 5
Unsupported claims    : 0 / 30
```

The results indicate strong response quality on the sampled examples.

However, the sample size is small and therefore these numbers should not be interpreted as production-level quality estimates.

---

# 20. Human vs LLM-Judge Agreement

An independent human audit was performed on:

```text
10 examples
```

The human reviewed:

- relevance,
- grounding,
- helpfulness,
- overall quality,
- unsupported claims.

The exact agreement with the LLM judge was:

| Dimension | Exact Agreement |
|---|---:|
| Relevance | 100% |
| Grounding | 80% |
| Helpfulness | 70% |
| Overall quality | 50% |
| Unsupported claim | 100% |

Mean absolute differences were:

| Dimension | MAD |
|---|---:|
| Relevance | 0.00 |
| Grounding | 0.20 |
| Helpfulness | 0.40 |
| Overall quality | 0.50 |
| Unsupported claim | 0.00 |

The human audit provides evidence about judge behavior but is too small to constitute a statistically strong validation of the LLM judge.

---

# 21. Failure Analysis

The final classifier was evaluated using 5-fold out-of-fold predictions.

Results:

```text
Golden examples       : 200
Misclassified         : 101
Classification error  : 50.5%
```

The main recurring failure boundaries are:

1. `icloud_backup` ↔ `display_input`
2. `connectivity` ↔ `apple_services`
3. `display_input` ↔ `battery_charging`
4. `ios_update` ↔ `connectivity`
5. `apple_services` ↔ `apps_app_store`

---

## Failure 1: iCloud Backup vs Display/Input

Example:

```text
I want my fucking picture back pull it up on iCloud...
```

True:

```text
icloud_backup
```

Predicted:

```text
display_input
```

### Hypothesis

The classifier may be influenced by generic device/photo vocabulary and fail to recognize that the primary issue is cloud-based recovery.

### Possible improvement

Add more examples involving:

- iCloud recovery,
- backup restoration,
- photo synchronization,
- missing iCloud data.

---

## Failure 2: Connectivity vs Apple Services

Example:

```text
you really need to fix this update or something because every time me
and my friend try to FaceTime or call, the other person doesn’t get any
notification...
```

True:

```text
connectivity
```

Predicted:

```text
apple_services
```

### Hypothesis

Terms such as FaceTime and calling strongly resemble Apple-service requests, even when the underlying failure is communication/connectivity behavior.

### Possible improvement

Add explicit boundary examples distinguishing:

```text
service functionality
```

from:

```text
network/connectivity failure
```

---

## Failure 3: Display/Input vs Battery

Example:

```text
The “done” button will not let me select my picture. It’s covered in the
top right corner of the phone with the battery and wifi icons.
```

True:

```text
display_input
```

Predicted:

```text
battery_charging
```

### Hypothesis

The classifier is distracted by the explicit word "battery", even though battery status is incidental to the actual UI problem.

### Possible improvement

Add more training examples where component/status words are incidental and the actual user action or failure defines the intent.

---

## Failure 4: iOS Update vs Connectivity

Example:

```text
Me eternally waiting for my iPhone to do ANYTHING after updating to
#iOS11 #SLOWAF
```

True:

```text
ios_update
```

Predicted:

```text
connectivity
```

### Hypothesis

The classifier has difficulty distinguishing update-triggered device behavior from a true connectivity problem.

### Possible improvement

Add more examples where:

```text
iOS version/update
```

is context rather than the actual issue, while adding clearer examples of update failures.

---

## Failure 5: Apple Services vs Apps/App Store

Example:

```text
I went on iTunes and re downloaded it and it still isn’t in my library.
```

True:

```text
apple_services
```

Predicted:

```text
apps_app_store
```

### Hypothesis

Terms related to downloading and applications overlap with App Store language, while iTunes is categorized under Apple services.

### Possible improvement

Add explicit boundary examples for:

- iTunes,
- Apple Music,
- iMessage,
- FaceTime,
- App Store.

A hierarchical classification approach could also be considered.

---

## General Failure Pattern

The errors are concentrated around semantically adjacent intents rather than completely unrelated categories.

The major boundaries are:

```text
ios_update       <-> display_input
connectivity     <-> display_input
apple_services   <-> connectivity
apple_services   <-> apps_app_store
```

This suggests that the major limitation is not simply model capacity.

The taxonomy itself contains overlapping language and requires reasoning about the customer's **primary failure mechanism**.

Detailed failure examples are available in:

```text
reports/failure_analysis.md
```

Machine-generated failure examples are available in:

```text
data/evaluation/failure_examples.csv
```

---

# 22. What Is Misleading About the Headline Number?

The headline:

```text
49.5% intent accuracy
```

should not be interpreted as the overall quality of the support agent.

There are several reasons.

## Small Evaluation Set

The classifier result is based on only:

```text
200 examples
```

across:

```text
12 intents
```

---

## Uneven Class Distribution

Some intents contain considerably fewer examples than others.

This is why macro F1 is also reported.

---

## Classification Is Only One Component

The complete system is:

```text
Intent classification
        +
Historical retrieval
        +
Response generation
        +
Escalation
```

Therefore intent accuracy alone cannot measure end-to-end support quality.

---

## Escalation Was Tuned During Development

The escalation threshold was selected using development results on the golden set.

Therefore observed escalation results are not independent production safety measurements.

---

## LLM Evaluation Is Small

The:

```text
4.77 / 5
```

overall-quality result comes from only:

```text
30 generated responses
```

---

## Human Audit Is Small

Human-vs-judge agreement was evaluated on only:

```text
10 examples
```

---

## Retrieval Diagnostic Is Not Independent Ground Truth

The retrieval intent-alignment diagnostic uses classifier predictions.

It is therefore not equivalent to independently human-labeled retrieval relevance.

---

# 23. Limitations

The current prototype has several limitations.

## 23.1 Small Golden Set

200 examples are insufficient to robustly estimate performance across all 12 intents.

---

## 23.2 Class Imbalance

Some intents have very limited representation.

For example, the golden set contains only a small number of:

```text
icloud_backup
other_unclear
```

examples.

---

## 23.3 Context Handling

Previous-message context is included when available, but the current classifier does not perform full conversational reasoning.

---

## 23.4 Multi-Issue Requests

A single customer message can contain several issues.

The current taxonomy forces a primary intent.

---

## 23.5 Escalation Calibration

The escalation threshold was tuned on development data.

A production implementation should use a separate validation set.

---

## 23.6 Retrieval Evaluation

Retrieval quality has not yet been independently human-labeled.

---

## 23.7 LLM Judge Validation

Only 10 examples were independently reviewed by a human.

A larger human-rated set would provide stronger evidence.

---

## 23.8 No Production Customer Actions

The system does not access real customer accounts or perform actions such as:

- refunds,
- password resets,
- account modifications,
- purchases,
- ticket updates.

---

# 24. Decision Log

Non-obvious decisions are documented in:

```text
reports/decision_log.md
```

The decision log covers:

1. AppleSupport brand selection.
2. Customer/support pair construction.
3. Golden-set exclusion from retrieval and training.
4. Previous-message context retention.
5. Response leakage prevention.
6. Compact 12-intent taxonomy.
7. `other_unclear` fallback.
8. Rejection of keyword-only classification.
9. Word + character TF-IDF + Linear SVM.
10. Historical semantic retrieval.
11. FAISS index caching.
12. Deterministic escalation.
13. Development confidence threshold.
14. LLM-as-judge evaluation.
15. Human audit of the judge.

---

# 25. Project Structure

```text
hiver-support-agent/
│
├── data/
│   ├── evaluation/
│   │   ├── apple_support_golden_200.csv
│   │   ├── apple_support_llm_evaluation_sample_30.csv
│   │   ├── apple_support_llm_judge_30.csv
│   │   ├── apple_support_human_review_10.csv
│   │   ├── evaluation_results.json
│   │   ├── failure_examples.csv
│   │   └── llm_judge_evaluation_results.json
│   │
│   ├── processed/
│   │   ├── apple_support_clean.csv
│   │   ├── apple_support_training_candidates.csv
│   │   └── retrieval_index/
│   │       ├── apple_support.faiss
│   │       └── retrieval_corpus.csv
│   │
│   └── raw/
│       └── twcs/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_intent_taxonomy.ipynb
│
├── reports/
│   ├── decision_log.md
│   └── failure_analysis.md
│
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── config.py
│   ├── data_processing.py
│   ├── escalation.py
│   ├── evaluation.py
│   ├── failure_analysis.py
│   ├── intent_classifier.py
│   ├── llm_judge.py
│   ├── retrieval.py
│   ├── run_evaluation.py
│   └── run_judge_evaluation.py
│
├── tests/
│   ├── test_data_processing.py
│   ├── test_escalation.py
│   ├── test_intent_classifier.py
│   └── test_retrieval.py
│
├── .env
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

---

# 26. Environment Setup

## Git LFS Prerequisite

The precomputed FAISS retrieval index is stored with Git LFS. Install Git LFS before cloning this repository:

```powershell
git lfs install
```

After cloning, download the LFS-managed files if they were not checked out automatically:

```powershell
git lfs pull
```

Python 3.11+ is recommended.

Create the virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

# 27. Dataset Setup

The raw Kaggle dataset is expected under:

```text
data/raw/twcs/
```

The primary file is:

```text
data/raw/twcs/twcs.csv
```

The dataset is not regenerated automatically by the evaluation commands.

After placing the dataset in the expected location, the notebooks can be used for the exploratory and preprocessing workflow.

The processed AppleSupport dataset is:

```text
data/processed/apple_support_clean.csv
```

---

# 28. Reproducing the Evaluation

## Intent Evaluation

Run:

```powershell
python -m src.run_evaluation
```

This evaluates:

- majority baseline,
- TF-IDF + Complement NB,
- TF-IDF + Linear SVM,
- TF-IDF + Logistic Regression,
- Word + Character TF-IDF + Linear SVM.

It also reports:

- accuracy,
- macro F1,
- weighted F1,
- per-intent precision/recall/F1,
- confusion pairs.

Results are saved to:

```text
data/evaluation/evaluation_results.json
```

---

## LLM Judge Evaluation

Run:

```powershell
python -m src.run_judge_evaluation
```

This evaluates the stored LLM-judge results and compares them with the 10-example human review.

It writes:

```text
data/evaluation/llm_judge_evaluation_results.json
```

This command does not make new Gemini requests.

---

# 29. Running Failure Analysis

Run:

```powershell
python -m src.failure_analysis
```

This generates 5-fold out-of-fold predictions and identifies misclassified examples with their decision margins.

It identifies:

- misclassified examples,
- classifier decision margins,
- lowest-confidence failures.

Output:

```text
data/evaluation/failure_examples.csv
```

The report interpretation is documented in:

```text
reports/failure_analysis.md
```

---

# 30. Running the Agent

The agent requires a Gemini API key.

Create:

```text
.env
```

with:

```env
GEMINI_API_KEY=your_api_key_here
```

Example:

```python
from src.agent import create_agent

agent = create_agent()

result = agent.run(
    customer_message="My iPhone battery is draining very quickly."
)

print("Intent:", result["intent"])
print("Margin:", result["classifier_margin"])
print("Similarity:", result["top_similarity"])
print("Reply:", result["draft_reply"])
print("Escalate:", result["escalate"])
print("Reason:", result["escalation_reason"])
```

The result contains:

```text
intent
classifier_margin
top_similarity
draft_reply
evidence_used
escalate
escalation_reason
retrieved_cases
```

---

## First Run

The first retrieval initialization may require encoding the historical AppleSupport corpus and building the FAISS index.

This can take several minutes.

The index is then cached under:

```text
data/processed/retrieval_index/
```

Subsequent runs load the cached index.

---

# 31. Running Tests

Run:

```powershell
pytest -q
```

Expected:

```text
10 passed
```

The tests cover:

- customer/support pair construction,
- intent classifier behavior,
- escalation rules,
- retrieval behavior.

The project also includes:

```text
pytest.ini
```

to make the `src` package importable during test execution.

---

# 32. Generated Artifacts

The project produces several evaluation artifacts.

## Intent evaluation

```text
data/evaluation/evaluation_results.json
```

## LLM judge evaluation

```text
data/evaluation/llm_judge_evaluation_results.json
```

## Failure examples

```text
data/evaluation/failure_examples.csv
```

## Retrieval index

```text
data/processed/retrieval_index/
```

The retrieval index is generated locally and is ignored by Git.

---

# 33. What Is Not Built

This project intentionally does not attempt to build a complete production support platform.

The project does not include:

- production web UI,
- production ticketing integration,
- real customer-account access,
- real refunds,
- password resets,
- account modifications,
- automated purchases,
- production authentication/authorization,
- full-dataset model training,
- production-grade safety evaluation,
- automatic execution of customer-support actions.

The focus is:

```text
Intent
  ↓
Historical Evidence
  ↓
Grounded Draft
  ↓
Escalation Decision
```

---

# 34. Next Week

If development continued beyond the take-home, the highest-priority work would be:

## 1. Expand the Golden Set

Increase the number of manually reviewed examples, especially for underrepresented intents.

---

## 2. Improve Intent Boundaries

Add deliberate examples around the most confused intent pairs:

```text
ios_update ↔ display_input
connectivity ↔ display_input
apple_services ↔ connectivity
apple_services ↔ apps_app_store
```

---

## 3. Improve Context-Aware Classification

Use conversational context more systematically for messages that depend on previous turns.

---

## 4. Separate Threshold Calibration

Create a dedicated validation set for escalation threshold selection rather than tuning the policy on the final evaluation set.

---

## 5. Independently Evaluate Retrieval

Create human relevance labels for retrieved historical cases.

This would provide a better measurement than using classifier predictions as a retrieval diagnostic.

---

## 6. Expand Human Response Evaluation

Increase the number of human-reviewed generated responses.

---

## 7. Validate the LLM Judge

Compare judge scores against a substantially larger human-rated dataset.

---

## 8. Improve Multi-Issue Handling

Investigate hierarchical or multi-label classification for customer messages containing several independent issues.

---

## 9. Improve Vague-Message Handling

Use conversational context and uncertainty signals to better handle messages that are individually ambiguous.

---

## 10. Production Safety Layer

Before allowing automated customer actions, add explicit authorization, policy validation, audit logging, and action-specific safety checks.

---

# 35. Summary

This project implements a retrieval-grounded AI support agent for AppleSupport.

The system combines:

```text
Supervised Intent Classification
              +
Historical Semantic Retrieval
              +
Grounded LLM Response Generation
              +
Deterministic Escalation
              +
Automated Evaluation
              +
LLM-as-Judge Evaluation
              +
Human Audit
```

The strongest tested intent classifier achieved:

```text
Accuracy    : 49.5%
Macro F1    : 42.7%
Weighted F1 : 46.8%
```

on a 200-example golden evaluation set.

The response-generation evaluation produced:

```text
Mean relevance        : 4.97 / 5
Mean grounding        : 5.00 / 5
Mean helpfulness      : 4.77 / 5
Mean overall quality  : 4.77 / 5
Unsupported claims    : 0 / 30
```

The project deliberately reports the limitations of these numbers rather than presenting them as production-quality guarantees.

The main engineering focus is the complete support-agent loop:

```text
Customer Request
       ↓
Intent
       ↓
Historical Support Evidence
       ↓
Grounded Response
       ↓
Auto-handle / Human Escalation
```

Detailed modeling decisions and failure analysis are available in:

```text
reports/decision_log.md
reports/failure_analysis.md
```
