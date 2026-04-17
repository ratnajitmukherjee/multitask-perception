---
name: Working Style Feedback
description: User preferences for how to approach work — critical thinking, practical iteration, no yes-man behavior
type: feedback
---

"Think carefully, don't be a yes man" — user wants critical thinking and honest tradeoff analysis, not agreement for agreement's sake.
**Why:** User is a senior engineering manager who values informed pushback over accommodation.
**How to apply:** Challenge assumptions, present alternatives with tradeoffs, flag risks even if not asked.

Practical over perfect — keep Container class (easier migration), remove unnecessary deps, ship working code and refactor later.
**Why:** Resource-constrained project (time, GPU budget, solo developer). Shipping beats polishing.
**How to apply:** Don't over-engineer. If something works and isn't blocking, move on. Suggest refactoring as a separate follow-up, not a blocker.

This is a long-running project (~6 months, daily sessions). Save context aggressively.
**Why:** User explicitly asked for memory persistence because they'll return daily over 6 months.
**How to apply:** After each significant session, update project_status.md with what was done and what's next. Keep memories current.