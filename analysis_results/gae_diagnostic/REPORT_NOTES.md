# GAE diagnostic report notes

## Required structure map

Title; technical conclusion; key findings with role-proxy and temporal-decay visuals; scope and definitions;
methodology; limitations and evidence availability; recommended telemetry and decision; further questions.

## Chart map

| Section | Question | Family | Fields | Supported claim | Palette |
|---|---|---|---|---|---|
| Role signal | How different is raw advantage energy by transition role and window? | Grouped bar | window, role, A2 proxy | Collision role carries materially larger raw advantage energy | Hard two-root comparator |
| Credit span | How strongly does lambda change backward TD-residual weight? | Multi-series line | seconds, relative weight, lambda | 0.99 sharply shortens 1-4 second credit versus 0.995 | Ordered categorical series |

## Caveat

The role chart uses an indirect pre-update value-loss proxy rather than persisted raw advantages. The line chart is a
deterministic sensitivity calculation, not observed performance. Browser QA may be structural-only if Chromium is absent.
