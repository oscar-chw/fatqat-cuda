---
title: "Job"
---

# Job

!!! warning "Job API under development"

    The Job API is under active development. Its interfaces and supported
    behavior may change between releases. Pin an exact FatQat version when
    reproducibility matters.

Native FATQAT backends and [`Estimator`][fatqat.Estimator] return a completed
[`Job`][fatqat.Job]. Call
[`result`][fatqat.Job.result] to obtain the result; it does not wait.

::: fatqat.Job
    options:
      members:
        - "status"
        - "result"
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
