"""MkDocs hook: the docstrings use a few Sphinx roles (:func:`x`); show them as plain code."""

import re

ROLE = re.compile(r":(?:func|class|meth|attr|data|mod|obj):(<code>.*?</code>)")


def on_page_content(html, **kwargs):
    return ROLE.sub(r"\1", html)
