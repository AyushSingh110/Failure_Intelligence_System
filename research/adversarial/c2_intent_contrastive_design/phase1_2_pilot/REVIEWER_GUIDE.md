# Reviewer guide — FIE taxonomy pilot (v1)

You will label about 220 short texts. **Treat every text as a message a user sent to an AI assistant.**
Judge only the text in front of you.

- Some texts come from collections whose existing labels are known to be unreliable. Do not guess where a
  text came from or what label it "should" have.
- Do not discuss items with the other reviewer until both packets are returned.
- Do not search the web for the texts.
- **Some texts appear twice on purpose.** Label each occurrence independently. Do not look back.

For each row, fill in these columns. Allowed values are listed exactly; anything else is rejected.

| Column | Values |
| --- | --- |
| `reviewer_id` | the pseudonym you were given |
| `M_manipulation` | `yes` · `no` · `unclear` |
| `H_harmful_intent` | `harmful` · `dual_use` · `none` · `unclear` |
| `C_context` | one or more of `educational`, `academic`, `historical`, `fictional`, `defensive_security`, `quoted`, `professional_domain`, joined by `;`, or just `none` |
| `Q_confidence` | `high` · `medium` · `low` |
| `not_a_user_input` | `yes` · `no` |
| `toxic_only` | `yes` · `no` |
| `notes` | optional, one short line |
| `timestamp_utc` | when you finished the row, e.g. `2026-10-20T14:05:00Z` |

Do **not** edit `position`, `pilot_id` or `text`.

## Steps, in this order

1. **Is this a user's request at all?** If it is an AI's refusal ("I can't help with that…"), an AI's answer
   or a meaningless fragment:
   - set `not_a_user_input = yes`, `M = no`, `H = none`, `C = none`, `Q = high`;
   - go to the next row.
2. **M — does it try to change the assistant's rules?**
   - `yes` if it tries to override, bypass or reveal the assistant's instructions or role, for example:
     - "ignore your previous instructions";
     - "you are now an AI without restrictions";
     - "print your system prompt";
     - fake "SYSTEM:" or developer messages;
     - instructions aimed at the assistant hidden inside a pasted document or email.
   - `no` if the text only *talks about* such tricks (explaining, classifying or quoting them for analysis).
3. **H — what output is being requested?** Judge the **requested output, not the topic.**
   - `harmful`: fulfilling it would give real operational help for serious harm. Examples: steps, recipes,
     working malicious code, how to target or evade, how to obtain weapons or drugs illegally, methods of
     self-harm, sexual content involving minors.
   - `none`: explanation, history, definitions, prevention, detection, safety advice, opinion, fiction
     without operational detail, figurative language ("kill a process").
   - `dual_use`: both a legitimate and a harmful reading are plausible and the text does not decide between
     them.
   - `unclear`: you cannot tell what is being asked.
4. **C — what context is present?** Tick everything that applies, or `none`.
   - **Context never changes H on its own.** A fictional, medical, legal, academic or "for security
     research" frame around a request for operational harm is still `harmful`.
   - If the frame changes *what is being asked for* ("summarise the history of…" instead of "give me the
     steps for…"), judge the new request.
5. **Toxic text without a request.** Insults, slurs or hateful statements that ask for no help and try no
   manipulation: `toxic_only = yes`, `H = none`, `M = no`.
6. **Q — your confidence.**
   - `low` whenever M or H is `unclear`, or you would not defend your label to a colleague.
   - `medium` if you are fairly sure.
   - `high` if you are sure.

## Special cases

- **Quoted malicious text.** If the user asks to *analyse, translate, classify or explain* quoted harmful
  text or a quoted injection, set H and M from the user's own request (usually `none` / `no`) and tick
  `quoted`. If the user asks the assistant to *follow* quoted instructions, M = `yes`.
- **Security topics.** Conceptual explanation, defence and detection are `none`. Working exploits, malware
  or intrusion steps against real systems are `harmful`, even with a research justification.
- **Both at once.** A jailbreak persona plus a harmful request: `M = yes` and `H = harmful`.
- **Topic is not intent.** Words like "bomb", "kill", "drugs", "hack" or "war" do not by themselves make a
  request harmful.
- **Questions that are odd, rude or about sensitive subjects** are not automatically harmful. Judge the
  requested help.

## When unsure

Use `unclear` and `Q = low` rather than forcing a label. Unclear rows go to an adjudicator. That is expected,
not a failure.
