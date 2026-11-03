"""Harness examples: middleware around every boundary call (guards,
verification, error recovery, telemetry).

    01_guarded_agent      allowlist, PII redaction, per-turn token budget,
                          schema/assert verifier with retry-with-feedback
    02_resilient_agent    retry / timeout / circuit breaker / fallback model,
                          ErrorBoundary, kill -9 + resume_pending
"""
