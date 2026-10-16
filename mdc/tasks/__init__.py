"""Deterministic compute kernels: the only code a worker ever executes.

Workers run nothing but the functions registered in
:data:`mdc.tasks.base.TASK_KINDS` (spec S.7.1): no code, bytecode, pickles or
module names ever travel over the network. All arithmetic is integer, so the
same input gives byte-identical output on x86 laptops and ARM phones — which
is what makes result verification by re-execution possible (Phase 10).

Standard library + numpy only (this package ships to phones).
"""
