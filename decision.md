# Decision Log

1. **Brand selection: AmazonHelp**  chosen over AppleSupport/Tesco/others based on evidence: highest reply volume (169,840 vs. next-highest 106,860), and lowest DM-deflection rate (0.6% vs. 30-50% for competitors), meaning replies are substantive and resolve issues in-thread rather than redirecting to DMs  critical for grounding replies in real resolutions.

2. **Language scope: English only**  ~94% of AmazonHelp's volume is English. Multi-lingual support is a real product concern but out of scope for this project; documented as an explicit limitation rather than silently ignored.

3. **Language filtering method: langdetect over ASCII-heuristic**  initial ASCII-ratio heuristic correctly filtered non-Latin scripts (Japanese) but let Latin-script languages (German, French) through undetected. Caught while manual inspection of sample outputs, not by trusting aggregate counts. Switched to the `langdetect` library for accurate detection.

4. **Subsampling: 8,000 conversation pairs (from 169,840 total)**  per the assignment's explicit guidance that a subsample is expected and encouraged. Used `random_state=42` for reproducibility. Running full-scale language detection on the complete dataset proved impractically slow (1hr+ with no completion), validating this as the correct approach, not just a shortcut.

5. **Intent taxonomy: 8 categories, defined inductively from real data**  Delivery Issue, Damaged/Defective Item, Refund/Return Request, Order Status/Tracking Inquiry, Account/Security Issue, Billing/Charge Issue, General Product/Service Question, Other/Unclear. Kept "Refund/Return" separate from "Delivery Issue" despite real-world overlap, since resolution patterns differ (refund vs. redelivery vs. replacement) this matters for RAG retrieval quality later. "Other/Unclear" included deliberately as an honest catch-all rather than forcing every message into a clean category.

6. Switched from Gemini to Groq API after hitting persistent free-tier rate limits (503 errors) that were being silently masked as 'Other/Unclear' classifications by our own fallback logic a good reminder that error-handling can hide real problems if not logged clearly. Also had to update from llama-3.3-70b-versatile (deprecated by Groq June 2026) to openai/gpt-oss-120b.

7. Switched from per-message to batched classification (15 messages/call) after hitting Groq's 30 RPM free-tier limit — necessary to keep the full pipeline reproducible within the assignment's 15-minute budget, not just a performance.

8. Reduced RAG knowledge-base corpus to ~900 pairs (from an original 8,000-pair sample) after encountering real API quota constraints chose to document and adapt to this limitation transparently rather than force a larger run that risked incomplete/corrupted results under deadline pressure. This size still provides meaningful coverage across all 8 intent categories for grounding replies.

9. Final RAG corpus: 300 classified pairs via Groq (openai/gpt-oss-120b), sequential batches of 15. Reduced from original 8,000-pair extraction after hitting three separate infrastructure walls (Groq token quota, Gemini's unusually restrictive new-project 20/day cap).

10. Used a local, open-source embedding model (all-MiniLM-L6-v2 via sentence-transformers) for RAG retrieval instead of another API after today's repeated quota issues, this was both a pragmatic and principled choice: zero rate limits, runs offline, and embedding is a well-suited task for a small local model rather than needing a large LLM.