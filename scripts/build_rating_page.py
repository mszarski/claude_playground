"""Fill deploy/rating_page.html with the rating items (scripts/rating_set.py) and their uploaded video URLs.

  python scripts/build_rating_page.py --items runs/rating/v1/items.json --urls runs/rating/urls.json \
      --out runs/rating/reachy_ratings.html

``--urls`` maps a video file name to its URL in the artifact's asset store. The page carries only
what raters see (the transcript, two earlier lines, the voice reading, Reachy's reading per clip) plus the model
name per side for the owner's results panel; MELD's label and the recipes stay out.
"""
import argparse
import json
import os

TEMPLATE = os.path.join(os.path.dirname(__file__), "..", "deploy", "rating_page.html")


def fix_text(t):
    """MELD's transcripts carry Windows-1252 punctuation as raw bytes (\x92 for an apostrophe): repair them."""
    try:
        return t.encode("latin-1").decode("cp1252")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--urls", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    urls = json.load(open(a.urls))
    items = []
    for it in json.load(open(a.items)):
        answers = it.get("answers") or [it["a"], it["b"]]          # k answers (or the first round's a / b)
        if not all(x["video"] in urls for x in answers):
            continue
        items.append({"id": it["id"], "pool": it["pool"], "context": [fix_text(c) for c in it["context"][-2:]],
                      "text": fix_text(it["text"]),
                      "voice": it["voice"],
                      "answers": [{"model": x["model"], "reading": x["reading"], "url": urls[x["video"]]}
                                  for x in answers]})
    html = open(TEMPLATE).read().replace("__ITEMS__", json.dumps(items).replace("</", "<\\/"))
    with open(a.out, "w") as f:
        f.write(html)
    print(f"{len(items)} comparisons -> {a.out}")


if __name__ == "__main__":
    main()
