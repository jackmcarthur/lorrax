"""Old page URLs that still work: each writes a small page that sends the reader on.

A renamed page keeps its old URL here, so links into the public manual from
before the rename still land.  The stub keeps any #anchor of the old link.
"""
import html
import os

#: {old page directory: new page directory}, both relative to the site root.
REDIRECTS = {
    "theory/shared-pole-w/": "theory/tangential-w/",   # 2026-10-21, tangential W model
}

_STUB = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Moved</title>
<link rel="canonical" href="{href}">
<meta http-equiv="refresh" content="0; url={href}">
<script>location.replace("{href}" + location.hash);</script>
</head><body><p>This page moved to <a href="{href}">{href}</a>.</p></body></html>
"""


def on_post_build(config, **kwargs):
    for old, new in REDIRECTS.items():
        href = html.escape(os.path.relpath(new, old).replace(os.sep, "/") + "/")
        path = os.path.join(config["site_dir"], old, "index.html")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(_STUB.format(href=href))
