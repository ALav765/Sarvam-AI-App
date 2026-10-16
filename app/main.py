import os, re, io, csv, json, sqlite3, uuid, datetime as dt
from typing import Optional, List
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

KEY = os.environ.get("SARVAM_API_KEY", "")
BASE = "https://api.sarvam.ai"
LLM = os.environ.get("SARVAM_LLM", "sarvam-30b")
TTS_MODEL = os.environ.get("SARVAM_TTS_MODEL", "bulbul:v3")
DB = os.environ.get("DB_PATH", "store.db")
TTS_LANGS = {"hi-IN","bn-IN","kn-IN","ml-IN","mr-IN","od-IN","pa-IN","ta-IN","te-IN","en-IN","gu-IN"}
HEAD = {"api-subscription-key": KEY, "Authorization": f"Bearer {KEY}"}
YES = {"yes","yeah","ok","okay","save","confirm","haan","han","haa","हाँ","हां","जी","avunu","అవును","ஆம்","aam","ಹೌದು","houdu","അതെ","ho","होय"}

# ---------------- DB ----------------
def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c

def init():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS settings(k TEXT PRIMARY KEY, v TEXT);
        CREATE TABLE IF NOT EXISTS customers(name TEXT PRIMARY KEY, balance_paise INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS transactions(id TEXT PRIMARY KEY, ts TEXT, type TEXT, customer TEXT,
          total_paise INTEGER, payment_mode TEXT, notes TEXT, lang TEXT, raw TEXT, confidence REAL);
        CREATE TABLE IF NOT EXISTS items(txn_id TEXT, name TEXT, qty REAL, unit TEXT, unit_price_paise INTEGER);
        CREATE TABLE IF NOT EXISTS conversations(ts TEXT, speaker TEXT, lang TEXT, original TEXT, translated TEXT);
        INSERT OR IGNORE INTO settings VALUES('owner_language','te-IN'),('customer_language','hi-IN'),('shop_name','My Store');
        """)
init()

def get_set(k):
    with db() as c: return c.execute("SELECT v FROM settings WHERE k=?", (k,)).fetchone()["v"]

def set_set(k, v):
    with db() as c: c.execute("INSERT OR REPLACE INTO settings VALUES(?,?)", (k, v))

# ---------------- Sarvam client ----------------
async def _post(path, **kw):
    async with httpx.AsyncClient(timeout=60) as cl:
        for attempt in range(2):
            r = await cl.post(BASE + path, headers=HEAD, **kw)
            if r.status_code < 500 and r.status_code != 429: break
        if r.status_code >= 400:
            print("SARVAM ERROR", path, r.status_code, r.text[:300])
            raise HTTPException(502, f"Sarvam {path} failed: {r.text[:200]}")
        return r.json()

async def stt(data: bytes, name="a.webm", lang="unknown"):
    j = await _post("/speech-to-text", files={"file": (name, data)},
                    data={"model": "saaras:v4", "language_code": lang})
    return j.get("transcript", ""), j.get("language_code") or lang

async def translate(text, src, tgt):
    if not text or src == tgt: return text
    j = await _post("/translate", json={"input": text[:2000], "source_language_code": src,
                    "target_language_code": tgt, "model": "sarvam-translate:v1"})
    return j["translated_text"]

async def tts(text, lang):
    lang = lang if lang in TTS_LANGS else "hi-IN"
    j = await _post("/text-to-speech", json={"text": text[:2500], "language_code": lang, "model": TTS_MODEL})
    return j["audios"][0]  # base64 wav

async def chat(messages, temperature=0.1):
    j = await _post("/v1/chat/completions", json={"model": LLM, "messages": messages, "temperature": temperature})
    txt = j["choices"][0]["message"]["content"] or ""
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()

def parse_json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m: raise ValueError("no json")
    return json.loads(m.group(0))

# ---------------- Models ----------------
class Item(BaseModel):
    name: str
    quantity: float = 1
    unit: str = "pcs"
    unit_price: Optional[float] = None

class Draft(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    type: str = "sale"  # sale|purchase|credit_given|credit_received|expense
    customer_name: Optional[str] = None
    items: List[Item] = []
    total_amount: float = 0
    payment_mode: str = "cash"  # cash|upi|card|credit|other
    notes: Optional[str] = None
    spoken_language: str = ""
    raw_transcript: str = ""
    confidence: float = 0.5

EXTRACT_PROMPT = """You are a careful bookkeeping assistant for a small Indian shop. Convert the owner's spoken
sentence (any Indian language, code-mixed, transcribed) into ONE JSON object, nothing else.
Schema: {"type":"sale|purchase|credit_given|credit_received|expense","customer_name":str|null,
"items":[{"name":str,"quantity":number,"unit":str,"unit_price":number|null}],"total_amount":number,
"payment_mode":"cash|upi|card|credit|other","notes":str|null,"confidence":0-1}
Rules: item names in English. Understand spoken amounts: "saade teen sau"=350, "do hazaar paanch sau"=2500,
"dedh sau"=150, lakh=100000, crore=10000000. Units: kilo/kg, seer, litre, dozen, packet, piece.
"udhaar/baaki/credit" => payment_mode credit (sale) or type credit_given. Never guess a missing amount or items:
if total_amount or items are missing/unclear, output {"clarify":"<short English question>"} instead.
If a PREVIOUS draft and a correction are given, apply the correction to the previous draft and output the full new object.
Examples:
"Ramesh ki 2 kilo chawal, 1 litre oil, 450 rupees, cash" -> {"type":"sale","customer_name":"Ramesh","items":[{"name":"rice","quantity":2,"unit":"kg","unit_price":null},{"name":"oil","quantity":1,"unit":"litre","unit_price":null}],"total_amount":450,"payment_mode":"cash","notes":null,"confidence":0.9}
"సురేష్‌కి రెండు డజన్ల గుడ్లు, మూడు వందలు, అప్పు" -> {"type":"sale","customer_name":"Suresh","items":[{"name":"eggs","quantity":2,"unit":"dozen","unit_price":null}],"total_amount":300,"payment_mode":"credit","notes":null,"confidence":0.85}
"Kumar paid 500 UPI for old balance" -> {"type":"credit_received","customer_name":"Kumar","items":[],"total_amount":500,"payment_mode":"upi","notes":"old balance","confidence":0.9}
"sold some rice" -> {"clarify":"What was the quantity and total amount?"}"""

async def extract(text, lang, prev: Optional[Draft] = None):
    user = text if not prev else f"PREVIOUS draft: {prev.model_dump_json()}\nCorrection: {text}"
    msgs = [{"role": "system", "content": EXTRACT_PROMPT}, {"role": "user", "content": user}]
    for _ in range(2):
        try:
            d = parse_json(await chat(msgs))
            if "clarify" in d: return None, d["clarify"]
            dr = Draft(**d, spoken_language=lang, raw_transcript=text)
            if prev: dr.id = prev.id
            if dr.total_amount <= 0: return None, "What was the total amount?"
            return dr, None
        except Exception as e:
            print("extract retry:", e)
    return None, "Sorry, I could not understand. Please say it again."

def readback(d: Draft):
    items = ", ".join(f"{i.quantity:g} {i.unit} {i.name}" for i in d.items) or d.type
    who = f"{d.customer_name}: " if d.customer_name else ""
    return f"{who}{items}, Rs {d.total_amount:g}, {d.payment_mode}. Save?"

async def say(text_en, lang):
    t = await translate(text_en, "en-IN", lang)
    return t, await tts(t, lang)

# ---------------- Persistence ----------------
def paise(x): return int(round((x or 0) * 100))

def save(d: Draft):
    with db() as c:
        if c.execute("SELECT 1 FROM transactions WHERE id=?", (d.id,)).fetchone(): return False  # idempotent
        c.execute("INSERT INTO transactions VALUES(?,?,?,?,?,?,?,?,?,?)", (d.id, dt.datetime.now().isoformat(timespec="seconds"),
            d.type, d.customer_name, paise(d.total_amount), d.payment_mode, d.notes, d.spoken_language, d.raw_transcript, d.confidence))
        for i in d.items:
            c.execute("INSERT INTO items VALUES(?,?,?,?,?)", (d.id, i.name, i.quantity, i.unit, paise(i.unit_price) if i.unit_price else None))
        delta = balance_delta(d.type, d.payment_mode, paise(d.total_amount))
        if d.customer_name and delta:
            c.execute("INSERT OR IGNORE INTO customers(name) VALUES(?)", (d.customer_name,))
            c.execute("UPDATE customers SET balance_paise=balance_paise+? WHERE name=?", (delta, d.customer_name))
    return True

def balance_delta(type_, mode, amt):
    if type_ == "credit_given" or (type_ == "sale" and mode == "credit"): return amt
    if type_ == "credit_received": return -amt
    return 0

def since(period):
    n = {"today": 0, "week": 6, "month": 29}.get(period, 0)
    return (dt.date.today() - dt.timedelta(days=n)).isoformat()

# ---------------- Query tools (parameterized SQL only) ----------------
def tool_summary(period="today", **_):
    with db() as c:
        r = c.execute("SELECT type, COUNT(*) n, SUM(total_paise) t FROM transactions WHERE ts>=? GROUP BY type", (since(period),)).fetchall()
    return {x["type"]: {"count": x["n"], "rupees": (x["t"] or 0) / 100} for x in r}

def tool_balance(name="", **_):
    with db() as c:
        r = c.execute("SELECT name, balance_paise FROM customers WHERE lower(name) LIKE ?", (f"%{name.lower()}%",)).fetchall()
    return [{"name": x["name"], "owes_rupees": x["balance_paise"] / 100} for x in r]

def tool_top(period="week", **_):
    with db() as c:
        r = c.execute("""SELECT i.name, SUM(i.qty) q, i.unit FROM items i JOIN transactions t ON t.id=i.txn_id
          WHERE t.ts>=? AND t.type='sale' GROUP BY i.name, i.unit ORDER BY q DESC LIMIT 5""", (since(period),)).fetchall()
    return [dict(x) for x in r]

def tool_recent(n=5, **_):
    with db() as c:
        r = c.execute("SELECT ts,type,customer,total_paise/100.0 rupees,payment_mode FROM transactions ORDER BY ts DESC LIMIT ?", (int(n),)).fetchall()
    return [dict(x) for x in r]

TOOLS = {"get_sales_summary": tool_summary, "get_customer_balance": tool_balance, "top_items": tool_top, "list_transactions": tool_recent}
QUERY_PROMPT = """Pick ONE tool for the shop owner's question. Output JSON only: {"tool":name,"args":{...}}.
Tools: get_sales_summary(period: today|week|month); get_customer_balance(name); top_items(period: today|week|month);
list_transactions(n). Customer names should be written in English letters."""

# ---------------- App ----------------
app = FastAPI(title="Store Assistant")

async def detect_text(t):
    try:
        j = await _post("/text-lid", json={"input": t[:1000]})
        return j.get("language_code")
    except Exception as e:
        print("text-lid failed", e); return None

def known(l):
    return l if l in TTS_LANGS else None

async def get_text(audio: Optional[UploadFile], text: Optional[str], lang="unknown"):
    """Returns (text, detected_language_or_None). Language is auto-detected for both voice and typed input."""
    if audio is not None:
        t, det = await stt(await audio.read(), audio.filename or "a.webm", lang)
        return t, known(det)
    t = (text or "").strip()
    return t, known(lang if lang != "unknown" else (await detect_text(t) if t else None))

def learn_owner(det):
    """Owner is the only speaker in Log/Ask modes, so their language is learned automatically."""
    if det: set_set("owner_language", det)
    return get_set("owner_language")

@app.get("/api/settings")
def settings():
    with db() as c: return {r["k"]: r["v"] for r in c.execute("SELECT * FROM settings")}

@app.post("/api/settings")
def settings_set(owner_language: Optional[str] = Form(None), customer_language: Optional[str] = Form(None), shop_name: Optional[str] = Form(None)):
    for k, v in dict(owner_language=owner_language, customer_language=customer_language, shop_name=shop_name).items():
        if v: set_set(k, v)
    return settings()

def log_conv(speaker, lang, o, t):
    with db() as c: c.execute("INSERT INTO conversations VALUES(?,?,?,?,?)", (dt.datetime.now().isoformat(timespec="seconds"), speaker, lang, o, t))

@app.post("/api/chat/customer")
async def chat_customer(audio: Optional[UploadFile] = File(None), text: Optional[str] = Form(None), lang: Optional[str] = Form(None)):
    """Customer speaks any language -> owner's language."""
    t, det = await get_text(audio, text, lang or "unknown")
    if not t: return {"original": "", "translated": "", "audio": None, "lang": det}
    if det and det != "unknown": set_set("customer_language", det)  # sticky
    cl, ol = get_set("customer_language"), get_set("owner_language")
    tr = await translate(t, cl, ol)
    log_conv("customer", cl, t, tr)
    return {"original": t, "translated": tr, "lang": cl, "audio": await tts(tr, ol)}

@app.post("/api/chat/owner")
async def chat_owner(audio: Optional[UploadFile] = File(None), text: Optional[str] = Form(None)):
    """Owner replies in own language -> customer's language."""
    ol, cl = get_set("owner_language"), get_set("customer_language")
    t, _ = await get_text(audio, text, ol)
    tr = await translate(t, ol, cl)
    log_conv("owner", ol, t, tr)
    return {"original": t, "translated": tr, "lang": cl, "audio": await tts(tr, cl)}

@app.post("/api/chat/auto")
async def chat_auto(audio: Optional[UploadFile] = File(None), text: Optional[str] = Form(None)):
    """Detects the language AND who is speaking: owner's language -> customer's language, anything else -> owner's."""
    t, det = await get_text(audio, text)
    if not t: raise HTTPException(400, "Nothing heard")
    ol, cl = get_set("owner_language"), get_set("customer_language")
    if det is None or det == ol:
        speaker, src, tgt = "owner", ol, cl
    else:
        speaker, src, tgt = "customer", det, ol
        set_set("customer_language", det)  # sticky until someone else speaks
    tr = await translate(t, src, tgt)
    log_conv(speaker, src, t, tr)
    return {"speaker": speaker, "original": t, "translated": tr, "from": src, "to": tgt, "audio": await tts(tr, tgt)}

async def draft_response(d, q, ol):
    if d is None:
        said, au = await say(q, ol); return {"draft": None, "clarify": said, "audio": au}
    said, au = await say(readback(d), ol)
    return {"draft": d.model_dump(), "readback": said, "audio": au}

@app.post("/api/log/parse")
async def log_parse(audio: Optional[UploadFile] = File(None), text: Optional[str] = Form(None)):
    t, det = await get_text(audio, text)
    if not t: raise HTTPException(400, "Nothing heard")
    ol = learn_owner(det)
    d, q = await extract(t, ol, None)
    return {"transcript": t, "lang": ol, **await draft_response(d, q, ol)}

@app.post("/api/log/correct")
async def log_correct(draft: str = Form(...), audio: Optional[UploadFile] = File(None), text: Optional[str] = Form(None)):
    """Owner voice reply to a pending draft: yes -> save, otherwise treat as correction."""
    prev = Draft(**json.loads(draft))
    t, det = await get_text(audio, text)
    ol = learn_owner(det)
    if t.strip(" .!?।").lower() in YES:
        save(prev); said, au = await say("Saved.", ol)
        return {"saved": True, "transcript": t, "readback": said, "audio": au}
    d, q = await extract(t, det, prev)
    return {"transcript": t, **await draft_response(d, q, ol)}

@app.post("/api/log/confirm")
def log_confirm(draft: Draft):
    return {"saved": True, "new": save(draft), "id": draft.id}

@app.post("/api/log/undo")
def log_undo():
    with db() as c:
        t = c.execute("SELECT * FROM transactions ORDER BY ts DESC, rowid DESC LIMIT 1").fetchone()
        if not t: return {"undone": False}
        delta = balance_delta(t["type"], t["payment_mode"], t["total_paise"])
        if t["customer"] and delta:
            c.execute("UPDATE customers SET balance_paise=balance_paise-? WHERE name=?", (delta, t["customer"]))
        c.execute("DELETE FROM items WHERE txn_id=?", (t["id"],)); c.execute("DELETE FROM transactions WHERE id=?", (t["id"],))
    return {"undone": True, "id": t["id"], "rupees": t["total_paise"] / 100}

@app.post("/api/query")
async def query(audio: Optional[UploadFile] = File(None), text: Optional[str] = Form(None)):
    t, det = await get_text(audio, text)
    ol = learn_owner(det)
    try:
        plan = parse_json(await chat([{"role": "system", "content": QUERY_PROMPT}, {"role": "user", "content": t}]))
        fn = TOOLS[plan["tool"]]; data = fn(**(plan.get("args") or {}))
    except Exception as e:
        print("query fail", e); data = None
    prompt = f"Owner asked: {t}\nData: {json.dumps(data)}\nAnswer in 1-2 short sentences in language code {ol}, amounts in rupees. If data is null say you could not find it."
    ans = await chat([{"role": "user", "content": prompt}], 0.3)
    return {"transcript": t, "data": data, "answer": ans, "audio": await tts(ans, ol)}

@app.get("/api/transactions")
def transactions(period: str = "month", customer: Optional[str] = None):
    q, a = "SELECT * FROM transactions WHERE ts>=?", [since(period)]
    if customer: q += " AND lower(customer) LIKE ?"; a.append(f"%{customer.lower()}%")
    with db() as c:
        rows = [dict(r) for r in c.execute(q + " ORDER BY ts DESC LIMIT 500", a)]
        for r in rows: r["items"] = [dict(i) for i in c.execute("SELECT name,qty,unit FROM items WHERE txn_id=?", (r["id"],))]
    return rows

@app.get("/api/summary")
def summary(period: str = "today"):
    with db() as c:
        bal = [dict(r) for r in c.execute("SELECT name, balance_paise/100.0 AS owes FROM customers WHERE balance_paise!=0 ORDER BY owes DESC")]
    return {"period": period, "totals": tool_summary(period), "top_items": tool_top(period), "credit_balances": bal}

@app.get("/api/export.csv")
def export():
    out = io.StringIO(); w = csv.writer(out); w.writerow(["time","type","customer","rupees","payment","items","transcript"])
    for r in transactions("month"):
        w.writerow([r["ts"], r["type"], r["customer"], r["total_paise"]/100, r["payment_mode"], "; ".join(f'{i["qty"]:g} {i["unit"]} {i["name"]}' for i in r["items"]), r["raw"]])
    return Response(out.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=transactions.csv"})

app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "static"), html=True))
