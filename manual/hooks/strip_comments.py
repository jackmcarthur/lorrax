"""Strip HTML comments from rendered pages.

The Markdown sources keep their hidden comments (source citations such as
<!-- src/gw/...:line --> and the DISCREPANCIES blocks) as the maintainers'
audit trail. Python-Markdown passes them through verbatim, so without this
hook they would ship in every published page's source.

on_page_content runs before the search plugin indexes the page and before
the theme template wraps it, so only Markdown-authored content is touched.
A comment written inside a code block is escaped (&lt;!--) by then and
survives as text.
"""

import re

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def on_page_content(html, page, config, files):
    return _COMMENT.sub("", html)
