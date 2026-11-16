#!/usr/bin/env python3
"""Export any annotate_cr.py packet as a self-contained HTML labelling page
(same design as e2/packet.html): the packet's own label set and instructions,
one radio group per item, translation shown beneath non-Latin originals when
the packet carries one, no network loads, content warning, CSV download.
The CSV it produces (item_id,label,ts) is converted back into the terminal
tool's <packet>.labels.csv by lib/html_labels_to_csv.py.

    python camera_ready/lib/packet_to_html.py camera_ready/e1/gateB_packet.json
"""
import json, sys, html, pathlib

pk = pathlib.Path(sys.argv[1])
P = json.loads(pk.read_text())
out = pk.with_suffix(".html")
labels = list(P["labels"].values())
N = len(P["items"])
key = pk.stem
def esc(s): return html.escape(s or "").replace("\n", "<br>")
rows = []
for k, it in enumerate(P["items"], 1):
    tr = f'<div class="resp"><b>MACHINE TRANSLATION</b> (for reference; the original is authoritative)<br>{esc(it["translation"])}</div>' if it.get("translation") else ""
    radios = "".join(f'<label><input type="radio" name="{it["item_id"]}" value="{l}"> {l}</label>' for l in labels)
    rows.append(f"""
<div class="item" id="item{k}">
<h3>Item {k} of {N}</h3>
<div class="req"><b>REQUEST</b><br>{esc(it['prompt'])}</div>
<div class="resp"><b>RESPONSE</b><br>{esc(it['text'])}</div>{tr}
<div class="lab">{radios}</div></div>""")
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(P['name'])}</title>
<style>
:root{{--bg:#fff;--fg:#14161C;--muted:#555;--box:#f4f4f2;--warn:#8a2a1a;--warnbg:#fbeeea}}
@media (prefers-color-scheme: dark){{:root{{--bg:#15171c;--fg:#e8e8e4;--muted:#aaa;--box:#22252c;--warn:#f0a090;--warnbg:#3a2420}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:860px;margin:0 auto;padding:16px}}
.warn{{background:var(--warnbg);color:var(--warn);border:1px solid var(--warn);padding:12px;border-radius:6px}}
.item{{border-top:1px solid #8884;padding:12px 0}} .req,.resp{{background:var(--box);padding:10px;border-radius:6px;margin:6px 0;white-space:normal}}
.lab label{{margin-right:18px}} button{{font:inherit;padding:8px 14px}} .sticky{{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid #8884}}
</style></head><body>
<div class="warn"><b>Content warning.</b> The requests below are harmful prompts from public safety benchmarks (AdvBench, HarmBench) and some responses comply with them. Do not redistribute this file. It loads nothing from the network and stores your answers only in this browser until you download them.</div>
<h1>{esc(P['name'])}</h1>
<p>You are labelling the <b>RESPONSE</b> to each request. The model, its configuration and every automated verdict are withheld. Do not try to guess or look up which model produced a response. Your progress is kept in this browser (local storage) until you press the download button, which saves a CSV you can send back.</p>
<h2>Instructions</h2><p>{esc(P['instructions'])}</p>
<div class="sticky"><span id="count">0 / {N} labelled</span> &nbsp; <button onclick="download()">download labels as CSV</button></div>
{''.join(rows)}
<div class="sticky" style="border-top:1px solid #8884;border-bottom:0"><span id="count2"></span> <button onclick="download()">download labels as CSV</button></div>
<script>
const N={N}, KEY={json.dumps(key)};
function labels(){{const o={{}};document.querySelectorAll('input[type=radio]:checked').forEach(r=>o[r.name]=r.value);return o;}}
function update(){{const n=Object.keys(labels()).length;document.getElementById('count').textContent=n+' / '+N+' labelled';document.getElementById('count2').textContent=n+' / '+N+' labelled';try{{localStorage.setItem(KEY,JSON.stringify(labels()));}}catch(e){{}}}}
document.querySelectorAll('input[type=radio]').forEach(r=>r.addEventListener('change',update));
try{{const s=JSON.parse(localStorage.getItem(KEY)||'{{}}');for(const k in s){{const el=document.querySelector('input[name="'+k+'"][value="'+s[k]+'"]');if(el)el.checked=true;}}}}catch(e){{}}
update();
function download(){{const l=labels();let csv='item_id,label,ts\\n';const ts=new Date().toISOString();for(const k in l)csv+=k+','+l[k]+','+ts+'\\n';const b=new Blob([csv],{{type:'text/csv'}});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=KEY+'_labels.csv';a.click();}}
</script></body></html>"""
out.write_text(page)
print(f"wrote {out} ({N} items, labels {labels})")
