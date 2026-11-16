"""Shared helpers for the CSET'26 camera-ready experiments.

Nothing here modifies anything outside camera_ready/. Existing modules are
imported from experiment/ (metrics.classify_v1, judge_refusals.RUBRIC, ...).
"""
import csv, datetime, hashlib, json, math, pathlib, random, sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESP = ROOT / "analysis" / "responses"
DATA = ROOT / "analysis" / "data"
CR = ROOT / "camera_ready"
sys.path.insert(0, str(ROOT / "experiment"))

LABELS = ("REFUSAL", "COMPLY", "GARBLED")

# ----------------------------------------------------------------- corpus
def cell_path(fam, cfg, suffix=".judged.json"):
    return RESP / f"{fam}__{cfg}{suffix}"

def load_cell(fam, cfg, suffix=".judged.json"):
    return json.loads(cell_path(fam, cfg, suffix).read_text())

def corpus_cells(include_zh=False):
    """(family, config) for every scored cell in the paper's corpus."""
    out = []
    for f in sorted(RESP.glob("*.judged.json")):
        fam, cfg = f.name[:-len(".judged.json")].split("__", 1)
        if fam.endswith("-zh") and not include_zh:
            continue
        out.append((fam, cfg))
    return out

def item_id(fam, cfg, idx, text):
    h = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:8]
    return f"{fam}__{cfg}__{idx}__{h}"

# -------------------------------------------------------- language detector
def is_nonlatin(t):
    """The paper's CJK/Latin detector (make_paper_figures.py `_nl`,
    make_paper_tables.py `_zh`): CJK share of alphabetic characters > 0.4."""
    cjk = sum(1 for c in t if "一" <= c <= "鿿")
    lat = sum(1 for c in t if c.isalpha() and ord(c) < 128)
    return cjk / max(cjk + lat, 1) > .4

def is_nonlatin_wide(t):
    """translate_ambiguous.py `nonlatin`: adds kana, hangul, Cyrillic, Arabic."""
    cjk = sum(1 for c in t if "一" <= c <= "鿿" or "぀" <= c <= "ヿ"
              or "가" <= c <= "힣")
    cyr = sum(1 for c in t if "Ѐ" <= c <= "ӿ")
    ara = sum(1 for c in t if "؀" <= c <= "ۿ")
    lat = sum(1 for c in t if c.isalpha() and ord(c) < 128)
    return (cjk + cyr + ara) / max(cjk + cyr + ara + lat, 1) > 0.4

# ------------------------------------------------------------- statistics
def kappa(a, b):
    """Cohen's kappa exactly as experiment/judge_shootout.py computes it."""
    labs = sorted(set(a) | set(b)); n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labs)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0

def agreement(a, b):
    return 100 * sum(x == y for x, y in zip(a, b)) / len(a)

def weighted_kappa(a, b, w):
    """Cohen's kappa with per-item weights (Horvitz-Thompson style)."""
    W = sum(w)
    labs = sorted(set(a) | set(b))
    po = sum(wi for x, y, wi in zip(a, b, w) if x == y) / W
    pa = {l: sum(wi for x, wi in zip(a, w) if x == l) / W for l in labs}
    pb = {l: sum(wi for y, wi in zip(b, w) if y == l) / W for l in labs}
    pe = sum(pa[l] * pb[l] for l in labs)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0

def weighted_agreement(a, b, w):
    W = sum(w)
    return 100 * sum(wi for x, y, wi in zip(a, b, w) if x == y) / W

def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return 100 * lo, 100 * hi

def binarise(v):
    return "REFUSAL" if v == "REFUSAL" else "COMPLY"

def confusion(gold, pred, labels=None):
    labels = labels or sorted(set(gold) | set(pred))
    m = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold, pred):
        m[g][p] += 1
    return m

# ----------------------------------------------------------- human labels
def human_labels():
    """The 132 reference labels with the 14 translation-based revisions applied,
    plus the pass each item came from (first 75 = pass 1, rest = pass 2; the
    file is append-only and timestamped, so order is pass order)."""
    rows = list(csv.DictReader(open(DATA / "human_labels.csv")))
    assert len(rows) == 132, len(rows)
    rev = {r["item_id"]: r for r in csv.DictReader(open(DATA / "human_labels_revised.csv"))}
    assert len(rev) == 14, len(rev)
    frame = frame_files()
    cache = {}
    out = []
    for k, r in enumerate(rows):
        key = (r["family"], r["config"])
        if key not in cache:
            cache[key] = json.loads(frame[key].read_text())["items"]
        it = cache[key][int(r["idx"])]
        assert item_id(r["family"], r["config"], int(r["idx"]), it["text"]) == r["item_id"]
        rec = dict(item_id=r["item_id"], family=r["family"], config=r["config"],
                   idx=int(r["idx"]), judge=r["judge"], keyword_v1=int(r["keyword_v1"]),
                   human_original=r["human"],
                   human=rev[r["item_id"]]["human_after"] if r["item_id"] in rev else r["human"],
                   revised=r["item_id"] in rev, seconds=float(r["seconds"]), ts=r["ts"],
                   pass_=1 if k < 75 else 2, prompt=it.get("prompt", ""), text=it["text"],
                   nonlatin=is_nonlatin(it["text"]))
        out.append(rec)
    return out

# ----------------------------------------------- the annotation sampling frame
ANNOT_START = datetime.datetime(2026, 9, 12, 23, 10)

def frame_files():
    """The scored 200-prompt cells that existed when annotation began.
    Cells later extended to 500 were snapshotted as *.judged200.json by
    merge_extension.py (copy2 keeps the original mtime); cells never extended
    are still their *.judged.json. Cells scored after annotation began (the
    INTG4 and -zh cells, 2026-09-14) were not in the frame."""
    frame = {}
    for f in RESP.glob("*.judged200.json"):
        fam, cfg = f.name[:-len(".judged200.json")].split("__", 1)
        frame[(fam, cfg)] = f
    for f in RESP.glob("*.judged.json"):
        fam, cfg = f.name[:-len(".judged.json")].split("__", 1)
        if (fam, cfg) not in frame:
            frame[(fam, cfg)] = f
    return {k: f for k, f in frame.items()
            if datetime.datetime.fromtimestamp(f.stat().st_mtime) < ANNOT_START}

def frame_pool():
    """Replicates experiment/annotate.py build_pool()/select()/select_cell() on
    the reconstructed frame and returns every frame item with its stratum and
    its inclusion probability under each pass. Raises if the replication does
    not reproduce the 132 labelled item ids exactly."""
    frame = frame_files()
    files = sorted(frame.values(), key=lambda f: f.name.replace(".judged200.json", ".judged.json"))
    dis, agr, comply = [], [], []
    allitems = {}
    for f in files:
        d = json.loads(f.read_text()); fam, cfg = d["family"], d["config"]
        for i, it in enumerate(d["items"]):
            t = it.get("text", "")
            rec = dict(item_id=item_id(fam, cfg, i, t), family=fam, config=cfg, idx=i,
                       judge=it["judge"], keyword_v1=int(bool(it["keyword_v1"])),
                       eligible=bool(t.strip()) and len(t) <= 2000)
            allitems[rec["item_id"]] = rec
            if not rec["eligible"]:
                rec["stratum"] = "ineligible"; continue
            kw = bool(it["keyword_v1"]); jr = it["judge"] == "REFUSAL"
            if kw != jr:
                dis.append(rec); rec["stratum"] = "disagree"
            elif it["judge"] == "COMPLY":
                comply.append(rec); rec["stratum"] = "comply"
            else:
                agr.append(rec); rec["stratum"] = "agree"
    rng = random.Random(0)
    for L in (dis, agr, comply):
        rng.shuffle(L)
    n = 75
    n_dis = min(len(dis), int(n * 0.60)); n_com = min(len(comply), max(6, int(n * 0.20)))
    n_agr = max(0, n - n_dis - n_com)
    sel = dis[:n_dis] + comply[:n_com] + agr[:n_agr]
    random.Random(1).shuffle(sel)
    p1 = {"disagree": n_dis / len(dis), "comply": n_com / len(comply), "agree": n_agr / len(agr)}
    for r in allitems.values():
        r["p1"] = p1.get(r["stratum"], 0.0)
    # pass 2: exhaustive non-COMPLY in the two central cells + 8 COMPLY filler each
    out2 = []
    for fam, cfg in [("Qwen2.5-7B-abl", "INT4"), ("Qwen2.5-7B-abl", "BF16")]:
        d = json.loads(frame[(fam, cfg)].read_text()); tgt, fill = [], []
        for i, it in enumerate(d["items"]):
            t = it.get("text", "")
            if not t.strip() or len(t) > 2000:
                continue
            rec = allitems[item_id(fam, cfg, i, t)]
            (tgt if it["judge"] != "COMPLY" else fill).append(rec)
        for r in tgt:
            r["p2"] = 1.0; r["stratum2"] = f"{fam}:{cfg}:nonCOMPLY"
        for r in fill:
            r["p2"] = 8 / len(fill); r["stratum2"] = f"{fam}:{cfg}:COMPLYfiller"
        r2 = random.Random(0); r2.shuffle(fill)
        out2 += tgt + fill[:8]
    random.Random(7).shuffle(out2)
    for r in allitems.values():
        r.setdefault("p2", 0.0); r.setdefault("stratum2", "none")
        r["pi"] = 1 - (1 - r["p1"]) * (1 - r["p2"])
    labs = list(csv.DictReader(open(DATA / "human_labels.csv")))
    ids1 = [r["item_id"] for r in sel]
    assert ids1 == [r["item_id"] for r in labs[:75]], "pass-1 replication failed"
    done = set(ids1)
    ids2 = [r["item_id"] for r in out2 if r["item_id"] not in done]
    assert ids2 == [r["item_id"] for r in labs[75:]], "pass-2 replication failed"
    return allitems, dict(n_frame=len(allitems), n_eligible=len(dis) + len(agr) + len(comply),
                          strata={"disagree": len(dis), "agree": len(agr), "comply": len(comply)},
                          draw={"disagree": n_dis, "agree": n_agr, "comply": n_com}, p1=p1,
                          n_cells=len(frame))

# ------------------------------------------------------------------ misc
def write_json(path, obj):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False))

def tex_escape(t):
    for a, b in [("\\", ""), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"),
                 ("$", "\\$"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}")]:
        t = t.replace(a, b)
    return t

def log_progress(msg):
    p = CR / "PROGRESS.md"
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(p, "a") as fh:
        fh.write(f"- {ts}  {msg}\n")
