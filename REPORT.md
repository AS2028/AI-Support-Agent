# AI Support Agent for AmazonHelp — Report

## 1. Problem Framing

**Brand chosen: AmazonHelp.** Selected using evidence, not preference: it has the highest reply
volume in the dataset (169,840 replies vs. 106,860 for the next-closest brand, AppleSupport), and
the lowest rate of deflecting customers to DMs (0.6% vs. 30-50% for competitors like AppleSupport
and Uber_Support). A low DM-deflection rate matters specifically for this task the reply-drafting
step needs real, in-thread resolutions to ground on, not "please DM us".

**What "good" means for this brand:**
- **Intent classification** is "good" if it reliably separates messages into categories that map
  to genuinely different resolution paths (e.g. a refund request needs different handling than a
  tracking question), not just categories that sound distinct.
- **Reply drafting** is "good" if it reflects Amazon's actual historical resolution style and
  content for similar issues not a generic, provider-agnostic customer-service tone.
- **Escalation** is "good" if it catches messages that carry real risk (account/security, billing,
  genuine anger) even at the cost of some unnecessary human review a missed escalation is worse
  than an unnecessary one.

**What we chose not to build:**
- **Multi-language support.** ~94% of AmazonHelp's traffic is English; the remaining ~6%
  (Japanese, German, French, etc.) was filtered out. A production system would need this; a
  take-home does not.
- **A learned/trained escalation classifier.** Escalation uses explicit, auditable rules instead of
  a third LLM call or a trained model. This was a deliberate choice: the assignment requires a
  *stated reason* for each decision, and rule-based logic can be defended precisely in an
  interview, whereas "the model decided" cannot.
- **Multi-turn conversation context.** Each message is classified/replied-to independently, not as
  part of a full conversation thread. Real support tickets often span multiple exchanges; this was
  out of scope given the time available.
- **A vector database.** Retrieval is done with in-memory cosine similarity over ~300 embeddings.
  At this scale a dedicated vector DB adds infrastructure without adding capability.

## 2. Results vs. Baselines

Evaluated on **164 hand-labeled golden examples** (see Section on the golden set below).

| Approach | Intent Accuracy |
|---|---|
| Trivial baseline (always predict most common intent) | 22.0% |
| Simple baseline (TF-IDF + Logistic Regression) | 43.9% |
| **Our system (LLM-based, few-shot)** | **61.0%** |

Our system nearly **triples** the trivial baseline and clearly beats the simple baseline. This
indicates the LLM-based approach is earning its additional cost and complexity, rather than
matching what a much cheaper classical model could already do.

**Escalation decision accuracy: 56.1%** (against 164 hand-labeled TRUE/FALSE judgments). This
number is discussed critically in Section 4, since it is close to chance level for a binary
decision.

**Reply quality (LLM-as-judge, 1-5 scale, n=25 sampled replies):**

| Criterion | Mean Judge Score |
|---|---|
| Relevance | ~3.8 |
| Tone | ~4.6 |
| Resolution | ~3.4 |

**Judge-human agreement** (the same 25 replies independently scored by a human):

| Criterion | Exact Match | Within 1 Point | Correlation |
|---|---|---|---|
| Relevance | 64.0% | 100.0% | 0.71 |
| Tone | 80.0% | 100.0% | 0.53 |
| Resolution | 32.0% | 84.0% | 0.39 |
| **Overall** | **58.7%** | **94.7%** | **0.68** |

The judge is reasonably trustworthy on relevance and tone, but weak on "resolution", the hardest
criterion, since judging whether a reply actually *moves toward solving* a problem requires
context (e.g., whether asking for a photo is genuinely useful) that text alone doesn't fully
provide.

## 3. Failure Analysis 

**1. "General Product/Service Question" ↔ "Other/Unclear" confusion (18 of 64 total errors —
the single largest failure pattern).**
Example: *"Hi, I'm currently unable to download books, though I restarted, updated, uninstalled
and reinstalled. Help?"* true label: General Product/Service Question; predicted: Other/Unclear.
**Hypothesis**: these two categories genuinely overlap for messages that are technical but mild
not quite a "problem" in the sense of Delivery/Damaged/Refund, but not purely informational either.
The taxonomy under-specifies this middle ground.

**2. Order Status/Tracking Inquiry vs. Delivery Issue overlap.**
Example: *"Not yet but status say not deliverable.. Order id: 402-8194592-8771514."*
**Hypothesis**: the line between "calmly asking" and "starting to complain" is genuinely blurry in
short text, and our definitions didn't give the classifier a sharp enough rule to separate them.

**3. Account/Security Issue is unpredictable only ~25% of true Account/Security messages were
correctly classified.**
Example: *"Regarding order 408-6370867-6223568. Please check the mail I have sent..."* true
label: Account/Security Issue; predicted: Order Status/Tracking Inquiry.
**Hypothesis**: partly a labeling issue on review, a few of the human "Account/Security" labels
in the golden set are themselves borderline calls, not unambiguous account/security problems. This
is flagged honestly rather than smoothed over (see Section 4).

**4. Escalation logic misses more real risk than it over-triggers.** Of 72 total escalation
mismatches: **41 were missed escalations** (customer needed a human, got auto-handled) vs. **31
over-escalations**.
Example: *"Tracking says pkg was delivered to mailroom. I don't have a mailroom nor my package."*
clearly an unresolved, frustrating situation, but contains none of the keyword-based frustration
signals (e.g. "unacceptable", "ridiculous") our rules look for.
**Hypothesis**: keyword-based frustration detection is too narrow; it catches explicit anger
language but misses calmly-worded messages describing a genuinely broken situation.

**5. Escalation accuracy (56.1%) is barely above chance for a binary decision.**
This is less a "pattern" than a summary finding: the escalation layer, as currently built, is the
weakest part of the system, and is discussed as the central caveat in Section 4.

## 4. What Is Misleading About My Headline Number?

The headline number most likely to be quoted from this project is **"61% intent accuracy, beating
baselines of 22% and 44%."** This is true, but incomplete in three important ways:

**1. The end-to-end system is much weaker than the intent number alone suggests.** A customer's
actual experience depends on classification *and* escalation working together. Escalation
accuracy is 56.1% barely better than a coin flip. Even where intent is classified correctly,
the decision of whether to let a customer see an auto-generated reply at all is unreliable. The
"61%" number describes only one stage of a two-stage pipeline whose weaker stage is closer to
random.

**2. The golden set itself has some labeling noise.** During failure analysis, several messages
labeled "Account/Security Issue" by the human annotator (this project's author, working alone,
under time pressure) don't unambiguously read as security issues on a second look. Since there
was no second annotator and no inter-rater agreement check on the *human* labels themselves (only
on the LLM-judge-vs-human check for reply quality), some fraction of the "wrong" predictions may
actually be inconsistent ground truth rather than genuine classifier errors. The true accuracy
could be a few points higher or lower than 61% depending on how much labeling noise exists.

**3. The intent distribution is naturally skewed, and accuracy hides per-class performance.**
"Other/Unclear" and "Delivery Issue" dominate the golden set (47 and 36 of 164 examples,
respectively), while "Damaged/Defective Item" has only 4. A single overall accuracy number is
pulled toward performance on the common classes; performance on rare-but-important classes like
Account/Security Issue is substantially worse than the headline
number implies, as shown directly in the failure analysis.

## 5. What I'd Do Next With One More Week

- **Fix the taxonomy boundary between "General Product/Service Question" and "Other/Unclear"**
  by either merging them or writing sharper, example-anchored definitions, this alone would
  likely resolve the single largest error pattern.
- **Replace keyword-based frustration detection** in escalation with a small LLM-based sentiment/
  urgency classifier, since the current approach demonstrably misses calm-but-serious complaints.
- **Expand the RAG corpus size** the current 300-pair corpus was constrained by real API quota
  limits encountered during development (see decision log), not by a considered choice about the
  right size; a larger, still-subsampled corpus (2,000-3,000 pairs) would likely improve grounding
  quality, especially for underrepresented intents like Damaged/Defective Item (only 4 examples in
  the golden set, likely similarly underrepresented in the RAG corpus).
- **Add conversation-level context** so multi-turn threads are handled as a unit rather than each
  message being classified independently.
- **Increase the judge-human agreement sample size** beyond 25 and specifically investigate why
  "resolution" scoring has the weakest agreement likely needs a more concrete
  rubric definition for what counts as "resolving" an issue.
