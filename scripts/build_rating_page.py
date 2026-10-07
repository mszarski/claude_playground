"""Fill deploy/rating_page.html with the rating items (scripts/rating_set.py) and their uploaded video URLs.

  python scripts/build_rating_page.py --items runs/rating/v1/items.json --urls runs/rating/urls.json \
      --out runs/rating/reachy_ratings.html

``--urls`` maps a video file name (``r000_a.mp4``) to its URL in the artifact's asset store. The page carries only
what raters see (the transcript, two earlier lines, the voice reading, Reachy's reading per clip) plus the model
name per side for the owner's results panel; MELD's label and the recipes stay out.
"""
import argparse
import json
import os

TEMPLATE = os.path.join(os.path.dirname(__file__), "..", "deploy", "rating_page.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--urls", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    urls = json.load(open(a.urls))
    items = []
    for it in json.load(open(a.items)):
        if not all(it[s]["video"] in urls for s in ("a", "b")):
            continue
        items.append({"id": it["id"], "pool": it["pool"], "context": it["context"][-2:], "text": it["text"],
                      "voice": it["voice"],
                      **{s: {"model": it[s]["model"], "reading": it[s]["reading"], "url": urls[it[s]["video"]]}
                         for s in ("a", "b")}})
    html = open(TEMPLATE).read().replace("__ITEMS__", json.dumps(items).replace("</", "<\\/"))
    with open(a.out, "w") as f:
        f.write(html)
    print(f"{len(items)} comparisons -> {a.out}")


if __name__ == "__main__":
    main()
