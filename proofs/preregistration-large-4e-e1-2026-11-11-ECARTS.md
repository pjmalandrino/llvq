# Departures from the E1 prereg of the large-4E budget

The prereg (`preregistration-large-4e-e1-2026-11-11.md`) is stamped and not edited. Departures, each written and
stamped before the job is submitted.

## E1, the spending cap (2026-11-11, before submission)

The operator raised the card spending cap from $15 to **$25**, relayed by the orchestrating session. **$10.31** is
spent; another session's job (6af4688d9737d9a9e092ebd9, large-prefill) adds at most **$0.96**. This job: about $0.54
(*estimated*), timeout 29 min, **$0.87** at worst, so **$12.14** of $25 at worst after it. Nothing else changes: the
arms, the setup, the predictions and the decision rule are the prereg's.

The image is published once that job's stage is RUNNING, COMPLETED or ERROR, from a clean detached checkout of the
commit that holds this file's stamp.
