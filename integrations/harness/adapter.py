"""How an ``llm`` boundary honors a verifier's retry (``rollout_id`` /
``feedback`` attrs, see ``pyreact.harness.verify``).

:func:`attempt_adapter` wraps the adapter in effect for one re-run of a
module: every predictor call of that run gets

- ``rollout_id`` in its LM kwargs (DSPy's way to bypass the LM cache for an
  otherwise identical request; stripped before the provider sees it), and
- with feedback, an extra input field ``feedback`` (renamed ``feedback_``
  if the signature already has one) holding the verifier's feedback --
  like ``dspy.Refine``'s ``hint_``. The module, its signature and the
  caller's inputs are untouched; only that attempt's prompts change.
"""

from __future__ import annotations

from typing import Any, Optional

import dspy

FEEDBACK_FIELD = "feedback"
FEEDBACK_DESC = ("Feedback on a previous attempt at this task (it was rejected): "
                 "fix these problems in your new answer")


def attempt_adapter(adapter: Any, *, feedback: Optional[str] = None,
                    rollout_id: Optional[int] = None) -> Any:
    """``adapter`` (a ChatAdapter, JSONAdapter, ...) conditioned for one retry.

    The returned object is an instance of a subclass of ``type(adapter)``
    with the same class name and state, so ``isinstance`` checks and
    streaming listeners (which look adapters up by class name) still work.
    """
    base = type(adapter)

    def _prep(lm_kwargs, signature, inputs):
        if rollout_id is not None:
            lm_kwargs = {**lm_kwargs, "rollout_id": rollout_id}
        if feedback:
            name = FEEDBACK_FIELD
            while name in signature.input_fields or name in signature.output_fields:
                name += "_"
            signature = signature.append(name, dspy.InputField(desc=FEEDBACK_DESC))
            inputs = {**inputs, name: feedback}
        return lm_kwargs, signature, inputs

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        lm_kwargs, signature, inputs = _prep(lm_kwargs, signature, inputs)
        return base.__call__(self, lm, lm_kwargs, signature, demos, inputs)

    async def acall(self, lm, lm_kwargs, signature, demos, inputs):
        lm_kwargs, signature, inputs = _prep(lm_kwargs, signature, inputs)
        return await base.acall(self, lm, lm_kwargs, signature, demos, inputs)

    cls = type(base.__name__, (base,), {"__call__": __call__, "acall": acall,
                                         "__module__": base.__module__})
    wrapped = cls.__new__(cls)
    wrapped.__dict__.update(adapter.__dict__)
    return wrapped
