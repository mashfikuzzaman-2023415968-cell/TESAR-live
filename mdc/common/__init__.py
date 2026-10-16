"""Code shared by the coordinator and the workers.

Everything here must run on Android/Termux with only the standard library and
numpy installed. The single exception is ``config.py`` (PyYAML), which only the
coordinator imports — see DECISIONS.md.
"""
