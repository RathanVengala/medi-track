"""
Local test + demo harness for MediTrack AI.

This file is not optional: the SpeechRecognition package installs its own
top-level `tests` package into site-packages, and a regular package always
wins over a namespace package during import resolution. Without this
__init__.py, `python -m tests.test_pipeline_local` resolves to SpeechRecognition's
tests directory and fails with "No module named tests.test_pipeline_local".
"""
