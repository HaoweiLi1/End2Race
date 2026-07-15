# B5-A opened-development regression evaluation

This is not fresh or final confirmation.

- Grid: 3 racelines x 4 speeds x 50 startpoints = 600
- BC (reused immutable B4 rows): collision=24, overtake=342, follow=234
- 95% overtake floor: 325

| variant | collision | overtake | follow | fixed | new | gained | lost | speed projection | feasible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| seed1_iter10 | 22 | 347 | 231 | 9 | 7 | 12 | 7 | 0 | True |
| seed1_iter20 | 25 | 349 | 226 | 5 | 6 | 12 | 5 | 0 | False |
| seed1_iter30 | 27 | 343 | 230 | 5 | 8 | 11 | 10 | 0 | False |

Selected: `seed1_iter10`
Verdict: **OPENED_DEVELOPMENT_SURVIVOR**
