# flake8: noqa: E501
"""Presentation layer for the Streamlit app (styling + HTML building blocks).

Nothing in here touches the model, the feature engineering or the Prometheus metrics -
it only renders things. HTML strings are written flush-left on purpose: Streamlit's
markdown parser turns indented lines into code blocks.
"""
from html import escape

FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');"

CSS = FONT_IMPORT + """
:root { --ink:#101114; --muted:#9a9ba3; --line:#e7e7ec; --card:#ffffff; --bg:#f6f6f9; --teal:#2f8f7b; --teal-d:#17483f; --red:#d64550; --amber:#d99a1e; }
html, body, .stApp, .stApp *:not([data-testid="stIconMaterial"]):not(svg):not(path) { font-family:'Manrope', system-ui, -apple-system, 'Segoe UI', sans-serif !important; }
.stApp { background: var(--bg); color: var(--ink); }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
.block-container { max-width: 1240px; padding-top: 1.2rem; padding-bottom: 2rem; }

/* ---------- hero ---------- */
.hero { position:relative; overflow:hidden; border-radius:26px; padding:46px 52px 38px; color:#fff;
  background: radial-gradient(circle at 12% 18%, rgba(66,200,170,.55) 0, rgba(66,200,170,0) 38%), radial-gradient(circle at 92% 90%, rgba(47,143,123,.35) 0, rgba(47,143,123,0) 40%), linear-gradient(135deg,#0c0d10 0%,#14181a 100%); }
.hero .logo { width:54px; height:54px; border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:26px;
  background:linear-gradient(145deg,#2b3033,#101214); border:1px solid rgba(255,255,255,.25); box-shadow:0 0 28px rgba(80,220,190,.35); margin-bottom:22px; }
.pill { display:inline-block; font-size:11px; font-weight:700; letter-spacing:.09em; text-transform:uppercase; padding:6px 12px; border-radius:999px; border:1px solid var(--line); background:#fff; color:#555; }
.hero .pill { background:rgba(255,255,255,.08); border-color:rgba(255,255,255,.22); color:#d8ebe6; }
.hero .h1 { color:#fff !important; font-size:46px; line-height:1.08; font-weight:800; letter-spacing:-.02em; margin:16px 0 10px; max-width:820px; }
.hero .h1 .m, .headline .m { color:#8b9296 !important; font-weight:700; }
.hero p { color:#a9b4b1; font-size:15.5px; max-width:640px; margin:0 0 26px; line-height:1.6; }
.stat-row { display:grid; grid-template-columns:repeat(5,1fr); gap:14px; }
.stat { background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.13); border-radius:18px; padding:16px 18px; backdrop-filter: blur(6px); }
.stat .k { font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:#8fa39e; font-weight:700; }
.stat .v { font-size:26px; font-weight:800; margin-top:6px; color:#fff; }
.stat .s { font-size:12px; color:#7e908c; margin-top:2px; }

/* ---------- sections ---------- */
.headline { color:var(--ink) !important; font-size:30px; font-weight:800; letter-spacing:-.015em; line-height:1.15; margin:26px 0 4px; color:var(--ink); }
.subtle { color:#767680; font-size:14px; margin-bottom:14px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:20px; padding:22px 24px; box-shadow:0 1px 2px rgba(16,17,20,.04); }
.card-title { font-size:13px; font-weight:800; letter-spacing:.02em; margin-bottom:2px; }
.chip { display:inline-block; font-size:12.5px; font-weight:600; padding:7px 13px; border-radius:999px; margin:0 8px 8px 0; }
.chip.bad { background:#fdecee; color:#a52632; border:1px solid #f6c9ce; }
.chip.good { background:#e8f5f1; color:#1f6b59; border:1px solid #bfe3d9; }

/* ---------- result ---------- */
.result { border-radius:24px; padding:30px 32px; color:#fff; margin-top:6px; }
.result.fraud { background: radial-gradient(circle at 90% 10%, rgba(255,120,120,.45) 0, rgba(255,120,120,0) 45%), linear-gradient(135deg,#2a1215,#14080a); }
.result.legit { background: radial-gradient(circle at 90% 10%, rgba(66,200,170,.45) 0, rgba(66,200,170,0) 45%), linear-gradient(135deg,#0f2a25,#08130f); }
.result .tag { font-size:11px; font-weight:800; letter-spacing:.1em; text-transform:uppercase; opacity:.8; }
.result .big { font-size:54px; font-weight:800; letter-spacing:-.03em; line-height:1.05; margin:6px 0 4px; }
.result .verdict { font-size:19px; font-weight:700; margin-bottom:18px; }
.meter { position:relative; height:12px; border-radius:999px; background:linear-gradient(90deg,#2f8f7b 0%,#d99a1e 55%,#d64550 100%); margin:22px 0 8px; }
.meter .dot { position:absolute; top:-5px; width:22px; height:22px; border-radius:50%; background:#fff; border:4px solid #101114; transform:translateX(-50%); box-shadow:0 2px 8px rgba(0,0,0,.4); }
.meter .thr { position:absolute; top:-9px; height:30px; width:2px; background:rgba(255,255,255,.85); }
.scale { display:flex; justify-content:space-between; font-size:11.5px; color:rgba(255,255,255,.65); }

/* ---------- misc ---------- */
.how { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }
.how .step { background:#fff; border:1px solid var(--line); border-radius:18px; padding:18px; }
.how .n { width:30px; height:30px; border-radius:50%; background:var(--teal-d); color:#fff; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:13px; margin-bottom:10px; }
.how h4 { margin:0 0 4px; font-size:15px; font-weight:800; }
.how p { margin:0; font-size:13px; color:#6e6f78; line-height:1.5; }
.trusted { display:flex; align-items:center; gap:14px; flex-wrap:wrap; margin:26px 0 8px; }
.trusted b { font-size:13px; color:#5c5d66; margin-right:6px; }
.tech { font-weight:800; font-size:15px; color:#b1b2ba; padding:4px 10px; }
.footer { margin-top:20px; border-radius:22px; background:#0f1113; color:#fff; padding:30px 36px; display:flex; justify-content:space-between; gap:30px; flex-wrap:wrap; }
.footer h3 { margin:0 0 6px; font-size:21px; font-weight:800; max-width:420px; line-height:1.25; color:#fff; }
.footer p { margin:0; color:#8d9391; font-size:13px; }
.footer .links a, .footer .links span { display:block; color:#cfd5d3; font-size:13px; margin-bottom:6px; text-decoration:none; }
.footer .links b { display:block; font-size:11px; letter-spacing:.09em; color:#7e8683; text-transform:uppercase; margin-bottom:8px; }

/* ---------- streamlit widgets ---------- */
div[data-testid="stForm"] { background:#fff; border:1px solid var(--line); border-radius:22px; padding:22px 26px; }
.stButton > button, div[data-testid="stFormSubmitButton"] > button { border-radius:999px; font-weight:700; border:1px solid var(--line); background:#fff; color:var(--ink); padding:.55rem 1.1rem; transition:all .15s ease; }
.stButton > button:hover { border-color:var(--teal); color:var(--teal-d); transform:translateY(-1px); box-shadow:0 6px 16px rgba(47,143,123,.15); }
div[data-testid="stFormSubmitButton"] > button { background:#101114; color:#fff; border-color:#101114; font-size:15px; padding:.75rem 1.2rem; }
div[data-testid="stFormSubmitButton"] > button:hover { background:var(--teal-d); border-color:var(--teal-d); color:#fff; }
button[data-baseweb="tab"] { font-weight:700; font-size:15px; }
div[data-baseweb="tab-highlight"] { background-color: var(--teal) !important; }
[data-testid="stSlider"] [role="slider"] { background-color: var(--teal) !important; }
@media (max-width: 900px) { .stat-row, .how { grid-template-columns:1fr 1fr; } .hero .h1 { font-size:32px; } .hero { padding:30px 24px; } }
"""


def hero(meta):
    m = meta["metrics"]
    stats = [
        ("Model", meta["best_model"], meta["training_method"]),
        ("ROC-AUC", f"{m['ROC-AUC']:.3f}", "ranking quality"),
        ("F1-Macro", f"{m['F1-Macro']:.3f}", "balanced accuracy of both classes"),
        ("Fraud recall", f"{m['Recall']:.0%}", "of real frauds caught"),
        ("Threshold", f"{meta['optimal_threshold']:.3f}", "tuned for best fraud F1"),
    ]
    cards = "".join(
        f'<div class="stat"><div class="k">{escape(k)}</div><div class="v">{escape(v)}</div><div class="s">{escape(s)}</div></div>'
        for k, v, s in stats)
    return (
        '<div class="hero"><div class="logo">🛡️</div>'
        '<span class="pill">UPI Fraud Shield · Live model</span>'
        '<div class="h1">Real-time fraud detection <span class="m">that works the way</span> your payments need it to</div>'
        '<p>Enter a UPI transaction and the machine-learning model scores its fraud risk in milliseconds, '
        'explains the warning signs and records every request for monitoring.</p>'
        f'<div class="stat-row">{cards}</div></div>')


def headline(dark, muted="", eyebrow=None, sub=None):
    out = ""
    if eyebrow:
        out += f'<div style="margin-top:26px"><span class="pill">{escape(eyebrow)}</span></div>'
    out += f'<div class="headline" style="margin-top:10px">{escape(dark)} <span class="m">{escape(muted)}</span></div>'
    if sub:
        out += f'<div class="subtle">{escape(sub)}</div>'
    return out


def card_title(text, sub=""):
    return f'<div class="card-title">{escape(text)}</div><div class="subtle" style="margin-bottom:6px">{escape(sub)}</div>'


def risk_signals(i):
    """Plain-language warning signs read from the raw inputs (display only - the model decides the verdict)."""
    avg = max(float(i["Avg_Transaction_Value"]), 1.0)
    return [
        ("Night-time transaction", i["Transaction_Hour"] in (0, 1, 2, 3, 4, 22, 23)),
        (f"Amount is {i['Transaction_Amount'] / avg:.1f}x the usual spend", i["Transaction_Amount"] > 3 * avg),
        ("New account (under 30 days)", i["Account_Age_Days"] < 30),
        ("KYC not verified", i["KYC_Status"] < 2),
        ("Device changed recently", bool(i["Device_Change_Flag"])),
        ("Suspicious device flagged", bool(i["Suspicious_Device"])),
        ("IP location mismatch", not i["IP_Location_Match"]),
        ("Rapid transactions flagged", bool(i["Rapid_Transaction_Flag"])),
        ("Many login attempts", i["Login_Attempts"] >= 4),
        ("Burst of payments in 24h", i["Transactions_Last_24h"] >= 10),
        ("Very little payment history", i["Previous_Transaction_Count"] < 10),
    ]


def chips(signals):
    hit = [s for s, on in signals if on]
    if not hit:
        return '<span class="chip good">No warning signs triggered</span>'
    return "".join(f'<span class="chip bad">{escape(s)}</span>' for s in hit)


def result_card(result, amount):
    p, thr = result["probability"], result["threshold"]
    fraud = result["is_fraud"]
    level = "High" if fraud else ("Medium" if p >= thr * 0.5 else "Low")
    verdict = "FRAUD SUSPECTED" if fraud else "Legitimate transaction"
    icon = "🚨" if fraud else "✅"
    return (
        f'<div class="result {"fraud" if fraud else "legit"}">'
        f'<div class="tag">{icon} Risk level: {level} · Amount ₹{amount:,.0f}</div>'
        f'<div class="big">{p:.1%}</div>'
        f'<div class="verdict">{verdict} — fraud probability {p:.1%}</div>'
        f'<div class="meter"><div class="thr" style="left:{thr * 100:.1f}%"></div><div class="dot" style="left:{min(p, 1) * 100:.1f}%"></div></div>'
        f'<div class="scale"><span>0% safe</span><span>decision threshold {thr:.3f}</span><span>100% fraud</span></div></div>')


def how_it_works(meta):
    steps = [
        ("Enter details", "Amount, time, account and device information of the UPI payment."),
        ("Engineer features", "18 risk features such as amount-to-average ratio, night flag and security score are created."),
        (f"{meta['best_model']} scores it", "The tuned model returns the probability that the payment is fraud."),
        ("Decide and monitor", "The tuned threshold gives the verdict, and Prometheus records the request, latency and class."),
    ]
    html = "".join(f'<div class="step"><div class="n">{n}</div><h4>{escape(t)}</h4><p>{escape(d)}</p></div>'
                   for n, (t, d) in enumerate(steps, 1))
    return f'<div class="how">{html}</div>'


def powered_by():
    names = ["CatBoost", "MLflow", "Docker", "Kubernetes", "Prometheus", "Grafana", "GitHub Actions"]
    return '<div class="trusted"><b>Powered by</b>' + "".join(f'<span class="tech">{n}</span>' for n in names) + "</div>"


def footer():
    return (
        '<div class="footer"><div><h3>Fraud detection with full MLOps monitoring built in</h3>'
        '<p>Every prediction is counted, timed and classified, ready for Grafana dashboards.</p></div>'
        '<div class="links"><b>Services</b><span>Streamlit UI · port 8501</span><span>Prometheus metrics · port 8000 /metrics</span>'
        '<span>Prometheus · 9090 &nbsp;|&nbsp; Grafana · 3000</span></div></div>')
