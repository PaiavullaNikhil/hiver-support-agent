# Decision Log

## 1. Selected AppleSupport as the target brand

**Decision:** Build the agent for `AppleSupport`.

**Why:** AppleSupport had enough historical support interactions while
offering a useful mix of technical troubleshooting, account issues,
connectivity problems, software updates, and device issues. This gives the
agent meaningful intent boundaries and historical responses to retrieve.

---

## 2. Used customer → AppleSupport response pairs as the core dataset

**Decision:** Construct training/retrieval examples from customer messages
paired with the AppleSupport response tweet.

**Why:** The assignment requires the agent to draft replies based on how the
brand historically resolved similar issues. The response paired with the
customer message provides direct evidence of the brand's historical support
behavior.

---

## 3. Excluded the golden set from model training

**Decision:** The 200 golden examples are excluded from the retrieval
corpus and are not used to train the final classifier.

**Why:** Reusing evaluation examples for retrieval or training would create
evaluation leakage and make the headline metrics less trustworthy.

---

## 4. Retained previous-message context when available

**Decision:** Preserve the previous tweet when the customer's message is a
reply to another tweet.

**Why:** Around 30% of the cleaned AppleSupport interactions had usable
preceding-message context. Short support messages such as "same issue" can
be impossible to classify correctly without the preceding conversation.

---

## 5. Did not use the brand response as classifier input

**Decision:** The historical AppleSupport response is not provided to the
intent classifier.

**Why:** The response often reveals the underlying issue and would create
target leakage. The classifier should predict the customer's intent from
the customer-side conversation only.

---

## 6. Defined a compact 12-intent taxonomy

**Decision:** Use 12 support intents rather than attempting to reproduce
every possible issue in the dataset.

**Why:** A small taxonomy makes the classifier and escalation policy
operationally manageable while still covering the major AppleSupport
problem categories observed in the sampled data.

---

## 7. Used `other_unclear` as an explicit fallback intent

**Decision:** Include `other_unclear` rather than forcing every message
into a specific technical category.

**Why:** Support messages can be vague, incomplete, or context-dependent.
Forcing uncertain examples into a specific class would produce misleading
confidence and increase the risk of inappropriate automation.

---

## 8. Rejected keyword-only intent classification

**Decision:** Keyword rules were used for taxonomy exploration but not as the
final classifier.

**Why:** Initial keyword analysis showed substantial overlap: many examples
matched multiple candidate intents while others matched none. Keyword
matching therefore provided useful exploratory signals but was not reliable
enough as the final classification mechanism.

---

## 9. Chose word + character TF-IDF with Linear SVM

**Decision:** The final intent classifier uses word and character TF-IDF
features with a balanced Linear SVM.

**Why:** This approach performed better than the tested simpler baselines
on the 200-example golden set and captures both semantic word patterns and
short/noisy social-media text patterns.

---

## 10. Used historical semantic retrieval rather than response templates

**Decision:** Retrieve similar historical customer-support cases using
sentence embeddings and FAISS.

**Why:** Historical responses contain the brand's actual support behavior.
Retrieval allows the LLM to synthesize a response from real examples rather
than relying entirely on generic model knowledge.

---

## 11. Cached the retrieval index

**Decision:** Persist the FAISS index and retrieval corpus under
`data/processed/retrieval_index/`.

**Why:** Encoding more than 100,000 historical messages takes several
minutes. Caching makes subsequent runs substantially faster and makes the
agent practical to reproduce locally.

---

## 12. Added deterministic escalation rules

**Decision:** Escalation uses classifier confidence, retrieval similarity,
vague-message detection, and specific account/device-related conditions.

**Why:** The LLM should not be the sole authority for deciding whether a
customer request can be safely automated. Deterministic signals make the
decision easier to inspect and reproduce.

---

## 13. Set the classifier-margin escalation threshold to 0.30

**Decision:** Use a decision-margin threshold of `0.30`.

**Why:** A threshold sweep on the golden set showed the trade-off between
automation rate and auto-handled accuracy. At 0.30, the development
evaluation produced a substantially safer auto-handling subset than lower
thresholds.

This threshold is treated as a development choice rather than a production
safety guarantee.

---

## 14. Used an LLM judge for response-quality evaluation

**Decision:** Evaluate generated replies using an LLM judge scoring
relevance, grounding, helpfulness, overall quality, and unsupported claims.

**Why:** Intent accuracy alone cannot determine whether the generated
customer response is useful or grounded in historical support behavior.
The additional rubric evaluates the actual support-agent output.

---

## 15. Added an independent human audit

**Decision:** Compare the LLM judge against human ratings on a 10-example
subset.

**Why:** An automated judge can have systematic biases. Comparing its scores
with an independent human review provides evidence about whether the judge
is measuring response quality consistently enough to be useful.