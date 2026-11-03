"""``llm_judge``: an LM-graded verification check."""

from __future__ import annotations

import asyncio
from typing import Any, Iterable, Optional

import dspy

from pyreact.core.harness import Boundary, through


def llm_judge(criteria: str, lm: Any = None, *, fields: Optional[Iterable[str]] = None,
              threshold: float = 0.7, name: str = "llm_judge"):
    """A check for :func:`pyreact.harness.verify`: an LM grades the result
    against ``criteria`` (score in [0, 1] + feedback; passes at
    ``threshold``). Reuses :func:`integrations.learn.metrics.llm_judge`, the
    metric GEPA uses, so a verifier and an optimizer can share one judge.

    ``lm``: the judge's LM (default: ``dspy.settings.lm``; pyreact apps
    configure LMs per call, so usually pass one). ``fields``: output fields
    to show the judge (default: all but reasoning/trajectory). The judge
    call is its own ``llm.judge`` boundary (nested under the judged call),
    run in a worker thread; its tokens are not counted by ``budget``.
    """
    from integrations.learn.metrics import llm_judge as judge_metric

    metric = judge_metric(criteria, lm=lm, fields=fields)

    async def judge_check(b: Boundary, result: Any):
        if lm is None and getattr(dspy.settings, "lm", None) is None:
            raise RuntimeError("llm_judge needs lm=... (no LM configured in dspy.settings)")
        example = dspy.Example(**b.inputs).with_inputs(*b.inputs)
        pred = result if isinstance(result, dspy.Prediction) else dspy.Prediction(result=result)

        async def run(jb: Boundary):
            score, feedback = await asyncio.to_thread(metric.score, example, pred)
            jb.attrs.update(score=score, feedback=feedback)
            return score, feedback

        return await through("llm.judge", name, {"criteria": criteria}, run)

    judge_check.check_name = name
    judge_check.threshold = threshold
    return judge_check
