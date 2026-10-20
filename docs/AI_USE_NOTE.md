# AI-use note

This note records what the AI assistant did and what the student decided. It will go into the report.

## Phase 0 (2026-10-20)

What the AI did:
- Read the student's RnD repository read-only (simulator code, logs, theory notes). Nothing was written there.
- Audited the saved simulation data (Section 2 of `docs/phase0/PHASE0_REPORT.md`).
- Collected the label rules from the code, with file:line references, and proposed exact definitions.
- Listed the student's proven conditions (gathered with the help of a sub-agent; line references spot-checked).
- Wrote small benchmark and pilot scripts (own code, `docs/phase0/bench/`) and measured cost on the cloud container.
- Drafted the generation plan, this note, `docs/VIVA_PREP.md` (skeleton) and `README.md`.

What the student decided:
- Project topic, toolbox and rules (`BRIEF.md`).
- 2026-10-20: use only the constant-circle law of the published CDC 2024 paper (Singh & Chakraborty),
  not the switching law. The AI then updated the Phase 0 report (sections U1-U6) and ran two small circle-law pilots.
- Pending (after the senior meeting, Wed Oct 21): the questions Q-A to Q-F in section U6 of the Phase 0 report,
  including the label numbers. No model has been built yet.
