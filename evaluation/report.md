# Evaluation Report

**Date:** 2026-06-17 23:26
**Mode:** hybrid
**Semantic threshold:** 0.7
**Model:** llama3.2:3b

## Summary

| Metric | Value |
|--------|-------|
| Total questions | 22 |
| Passed | 10 |
| Overall pass rate | 45% |
| Avg keyword match | 0.3241 |
| Avg semantic similarity | 0.7086 |
| Source page accuracy | 0.8333 |
| Citation verification rate | 0.8 |
| Unsupported accuracy | 1.0 |
| Avg latency | 12.22s |

## Per-Question Results

| ID | Type | Lang | Pass | Sem | KW | Pages | Cit | ReqKW | Latency |
|---|---|---|---|---|---|---|---|---|---|
| en-001 | factual | english | F | 0.94 | 1.00 | Y | 0.00 | Y | 8.09s |
| en-002 | factual | english | P | 0.72 | 0.25 | Y | 1.00 | Y | 8.61s |
| en-003 | factual | english | F | 0.59 | 0.15 | N | 1.00 | N | 5.49s |
| en-004 | factual | english | F | 0.67 | 0.33 | Y | — | N | 11.83s |
| en-005 | factual | english | F | 0.49 | 0.08 | Y | — | N | 9.98s |
| en-006 | methodology | english | F | 0.74 | 0.29 | Y | 1.00 | N | 8.88s |
| en-007 | comparison | english | P | 0.82 | 0.33 | Y | — | Y | 10.76s |
| ar-001 | factual | arabic | P | 0.81 | 0.75 | Y | — | Y | 10.84s |
| ar-002 | factual | arabic | F | 0.64 | 0.17 | Y | 1.00 | Y | 9.22s |
| ar-003 | factual | arabic | F | 0.72 | 0.23 | N | — | Y | 7.07s |
| ar-004 | factual | arabic | P | 0.74 | 0.12 | Y | — | Y | 8.91s |
| ar-005 | factual | arabic | P | 0.73 | 0.23 | Y | — | Y | 23.34s |
| mixed-001 | factual | mixed | F | 0.46 | 0.00 | Y | — | N | 67.8s |
| mixed-002 | comparison | mixed | P | 0.83 | 0.38 | Y | — | Y | 10.03s |
| mixed-003 | factual | mixed | F | 0.64 | 0.08 | Y | — | Y | 12.58s |
| mixed-004 | factual | mixed | F | 0.67 | 0.81 | N | — | Y | 27.82s |
| unsupported-001 | unsupported | english | P | — | — | — | — | — | 0.03s |
| unsupported-002 | unsupported | english | P | — | — | — | — | — | 4.29s |
| unsupported-003 | unsupported | english | P | — | — | — | — | — | 0.03s |
| unsupported-004 | unsupported | arabic | P | — | — | — | — | — | 0.03s |
| en-008 | methodology | english | F | 0.79 | 0.21 | Y | — | N | 5.85s |
| ar-006 | methodology | arabic | F | 0.77 | 0.40 | Y | — | N | 17.32s |
