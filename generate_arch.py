from PIL import Image, ImageDraw, ImageFont
import os, sys

W, H = 2400, 1700
BG       = "#0D1117"
PANEL    = "#161B22"
BORDER   = "#30363D"

# accent colours
C_FRONT  = "#1F6FEB"   # blue  – frontend
C_API    = "#388BFD"   # light blue – api gateway
C_CORE   = "#8B949E"   # grey  – core
C_PIPE   = "#3FB950"   # green – pipeline
C_EXT    = "#D29922"   # amber – external
C_DB     = "#BC8CFF"   # purple – DB
C_TEXT   = "#C9D1D9"
C_HEAD   = "#FFFFFF"
C_SUB    = "#8B949E"

def px(v): return int(v)

img  = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(img)

# ── font helpers ─────────────────────────────────────────────────────────────
def load(size):
    for name in ["arialbd.ttf","Arial Bold.ttf","DejaVuSans-Bold.ttf",
                 "DejaVuSans.ttf","arial.ttf","LiberationSans-Bold.ttf"]:
        for root in [r"C:\Windows\Fonts", "/usr/share/fonts/truetype/dejavu",
                     "/usr/share/fonts","."]:
            p = os.path.join(root, name)
            if os.path.exists(p):
                try: return ImageFont.truetype(p, size)
                except: pass
    return ImageFont.load_default()

F_TITLE = load(46)
F_HEAD  = load(26)
F_BODY  = load(19)
F_SMALL = load(16)
F_TINY  = load(13)

def rect(x1,y1,x2,y2, fill=PANEL, outline=BORDER, r=10, lw=2):
    draw.rounded_rectangle([x1,y1,x2,y2], radius=r, fill=fill, outline=outline, width=lw)

def text_c(cx,cy, txt, font=F_BODY, color=C_TEXT):
    bb = font.getbbox(txt)
    w,h = bb[2]-bb[0], bb[3]-bb[1]
    draw.text((cx - w//2, cy - h//2), txt, font=font, fill=color)

def text_l(x,y, txt, font=F_BODY, color=C_TEXT):
    draw.text((x, y), txt, font=font, fill=color)

def arrow(x1,y1,x2,y2, col=C_PIPE, lw=3):
    draw.line([(x1,y1),(x2,y2)], fill=col, width=lw)
    # arrowhead pointing toward (x2,y2)
    import math
    angle = math.atan2(y2-y1, x2-x1)
    sz = 14
    for a in [angle+2.5, angle-2.5]:
        draw.line([(x2,y2),(int(x2-sz*math.cos(a)), int(y2-sz*math.sin(a)))],
                  fill=col, width=lw)

def badge(cx,cy, txt, color, font=F_TINY):
    bb = font.getbbox(txt)
    tw,th = bb[2]-bb[0], bb[3]-bb[1]
    pw,ph = 14, 7
    draw.rounded_rectangle([cx-tw//2-pw, cy-th//2-ph,
                             cx+tw//2+pw, cy+th//2+ph],
                            radius=6, fill=color+"33", outline=color, width=1)
    draw.text((cx-tw//2, cy-th//2), txt, font=font, fill=color)

# ─────────────────────────────────────────────────────────────────────────────
#  TITLE
# ─────────────────────────────────────────────────────────────────────────────
draw.text((W//2 - 260, 22), "METRON  Platform", font=F_TITLE, fill=C_HEAD)
draw.text((W//2 - 280, 78), "High-Level Architecture  ·  AI LLM Evaluation & Security Testing",
          font=F_BODY, fill=C_SUB)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – FRONTEND  (top band)
# ─────────────────────────────────────────────────────────────────────────────
FY1, FY2 = 120, 260
rect(40, FY1, W-40, FY2, fill="#0D1F3C", outline=C_FRONT, r=14, lw=2)
text_l(60, FY1+8, "Frontend  ·  metron-ai  (Next.js 14 / TypeScript / Tailwind)", font=F_HEAD, color=C_FRONT)

pages = [
    ("Login / Register", "#1F6FEB"),
    ("Dashboard\n(Project Hub)", "#1F6FEB"),
    ("Configure\n(LLM + Adapters)", "#388BFD"),
    ("Builder\n(Module Selector)", "#388BFD"),
    ("Preview\n(Persona & Run)", "#388BFD"),
    ("Run\n(Live Pipeline)", "#3FB950"),
    ("Results\n(Functional/Security\nRAG/Perf/Load/Quality)", "#D29922"),
    ("Analysis\n(Compare Runs)", "#8957E5"),
]

n   = len(pages)
PW  = (W - 80 - (n-1)*14) // n
PY1 = FY1 + 46
PY2 = FY2 - 12

for i,(label, col) in enumerate(pages):
    px1 = 40 + i*(PW+14)
    px2 = px1 + PW
    draw.rounded_rectangle([px1,PY1,px2,PY2], radius=8,
                            fill=col+"22", outline=col, width=1)
    lines = label.split("\n")
    ly = PY1 + (PY2-PY1)//2 - (len(lines)*20)//2
    for ln in lines:
        bb = F_TINY.getbbox(ln)
        lw2 = bb[2]-bb[0]
        draw.text((px1+(PW-lw2)//2, ly), ln, font=F_TINY, fill=col)
        ly += 20

# ─────────────────────────────────────────────────────────────────────────────
# ARROW  Frontend → API
# ─────────────────────────────────────────────────────────────────────────────
arrow(W//2, FY2, W//2, FY2+50, col=C_API)
text_c(W//2+90, FY2+26, "REST API / SSE (FastAPI)", font=F_SMALL, color=C_API)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – BACKEND outer frame
# ─────────────────────────────────────────────────────────────────────────────
BY1, BY2 = FY2+50, 1540
rect(40, BY1, W-40, BY2, fill="#0A1F0A", outline=C_PIPE, r=14, lw=2)
text_l(60, BY1+8, "Backend  ·  metron-unified  (Python / FastAPI)", font=F_HEAD, color=C_PIPE)

# ── 2a  Core Infrastructure ──────────────────────────────────────────────────
CY1, CY2 = BY1+46, BY1+185
rect(60, CY1, W-60, CY2, fill="#161B22", outline=C_CORE, r=10, lw=1)
text_l(80, CY1+6, "Core Infrastructure", font=F_BODY, color=C_CORE)

core_boxes = [
    ("fastapi_server.py\n(API Gateway)", C_API, 80),
    ("llm_client.py\n(LLM Wrapper)", C_CORE, 340),
    ("config.py\n(Settings / Env)", C_CORE, 600),
    ("SQLite DB\n(db.py – runs / jobs)", C_DB, 860),
    ("models.py\n(Pydantic schemas)", C_CORE, 1120),
    ("deepeval_azure.py\n(Azure LLM Judge)", "#FF7B72", 1380),
]
CBW = 250; CBH = 85
for (lbl, col, cx) in core_boxes:
    bx1 = cx; bx2 = cx+CBW
    by1 = CY1+30; by2 = by1+CBH
    if bx2 > W-80: continue
    draw.rounded_rectangle([bx1,by1,bx2,by2], radius=8,
                            fill=col+"22", outline=col, width=1)
    lines = lbl.split("\n")
    lcy = by1+(CBH-len(lines)*20)//2
    for ln in lines:
        bb = F_TINY.getbbox(ln)
        draw.text((bx1+(CBW-bb[2]+bb[0])//2, lcy), ln, font=F_TINY, fill=col)
        lcy += 20

# Adapters sub-box
ax1,ax2 = 1650, 2320
ay1,ay2 = CY1+22, CY2-10
draw.rounded_rectangle([ax1,ay1,ax2,ay2], radius=8,
                        fill="#8957E5"+"22", outline="#8957E5", width=1)
text_c((ax1+ax2)//2, ay1+18, "Adapters", font=F_SMALL, color="#8957E5")
adapters = ["ChatbotAdapter","RAGAdapter","FormAdapter","MultiAgentAdapter"]
aw = (ax2-ax1-20)//2; apad = 5
for i,a in enumerate(adapters):
    r2=i//2; c2=i%2
    abx1 = ax1+10+c2*(aw+apad); abx2=abx1+aw
    aby1 = ay1+36+r2*38; aby2=aby1+30
    draw.rounded_rectangle([abx1,aby1,abx2,aby2], radius=5,
                            fill="#8957E5"+"44", outline="#8957E5", width=1)
    bb = F_TINY.getbbox(a); tw=bb[2]-bb[0]
    draw.text((abx1+(aw-tw)//2, aby1+7), a, font=F_TINY, fill="#D2A8FF")

# ── 2b  PIPELINE ─────────────────────────────────────────────────────────────
PL_Y1, PL_Y2 = CY2+18, BY2-18
rect(60, PL_Y1, W-60, PL_Y2, fill="#0D1117", outline=C_PIPE+"88", r=10, lw=1)
text_l(80, PL_Y1+6, "Evaluation Pipeline  (pipeline.py)", font=F_BODY, color=C_PIPE)

stages = [
    ("S0\nProfile",
     ["architecture_parser.py","document_parser.py"],
     "#79C0FF"),
    ("S1\nPersonas",
     ["persona_builder.py","fishbone_builder.py","coverage_validator.py"],
     "#56D364"),
    ("S2\nTests",
     ["security_gen.py","functional_gen.py","golden_dataset.py","quality_criteria.py"],
     "#F78166"),
    ("S3\nExecution",
     ["conversation_runner.py"],
     "#FFA657"),
    ("S4\nEvaluation",
     ["security.py","functional.py","rag.py","load.py","performance.py","quality.py"],
     "#D29922"),
    ("S5\nAggregation",
     ["aggregator.py"],
     "#BC8CFF"),
    ("S6\nFeedback",
     ["feedback_loop.py"],
     "#EC6547"),
    ("S7\nReport",
     ["report_generator.py"],
     "#3FB950"),
    ("S8\nRCA",
     ["prompt_classifier.py","rca_mapper.py"],
     "#FF7B72"),
]

NS = len(stages)
SY1 = PL_Y1+40; SY2 = PL_Y2-20
SW = (W - 120 - (NS-1)*12) // NS

for i,(title, files, col) in enumerate(stages):
    sx1 = 60 + i*(SW+12); sx2 = sx1+SW
    draw.rounded_rectangle([sx1,SY1,sx2,SY2], radius=10,
                            fill=col+"18", outline=col, width=2)
    # Stage header
    lines = title.split("\n")
    hy = SY1+10
    for ln in lines:
        bb = F_SMALL.getbbox(ln)
        draw.text((sx1+(SW-bb[2]+bb[0])//2, hy), ln, font=F_SMALL, fill=col)
        hy += 22
    # divider
    draw.line([(sx1+10, hy+2),(sx2-10,hy+2)], fill=col+"66", width=1)
    hy += 10
    # files
    for f in files:
        bb = F_TINY.getbbox(f)
        fw = bb[2]-bb[0]
        draw.text((sx1+(SW-fw)//2, hy), f, font=F_TINY, fill=C_TEXT)
        hy += 18
    # arrow to next stage
    if i < NS-1:
        ax = sx2+6; ay = (SY1+SY2)//2
        arrow(ax-4, ay, ax+8, ay, col=col, lw=2)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – EXTERNAL SERVICES
# ─────────────────────────────────────────────────────────────────────────────
EY1, EY2 = BY2+20, BY2+130
rect(40, EY1, W-40, EY2, fill="#1C1300", outline=C_EXT, r=12, lw=2)
text_l(60, EY1+6, "External Services & Data Sources", font=F_HEAD, color=C_EXT)

ext = [
    ("Target LLM Endpoint\n(User-supplied chatbot / API)", C_EXT),
    ("Azure OpenAI\n(LLM Judge via deepeval_azure)", "#FF7B72"),
    ("HuggingFace Datasets\n(AdvBench / HarmBench golden sets)", "#56D364"),
    ("Locust\n(Load / Performance testing)", "#FFA657"),
    ("RAGAS\n(RAG evaluation metrics)", "#79C0FF"),
    ("DeepEval\n(Functional eval metrics)", "#BC8CFF"),
]
NE = len(ext); EW = (W-80-(NE-1)*12)//NE
for i,(lbl,col) in enumerate(ext):
    ex1=40+i*(EW+12); ex2=ex1+EW
    ey1=EY1+34; ey2=EY2-10
    draw.rounded_rectangle([ex1,ey1,ex2,ey2], radius=7,
                            fill=col+"22", outline=col, width=1)
    lines=lbl.split("\n")
    lcy=ey1+(ey2-ey1-len(lines)*18)//2
    for ln in lines:
        bb=F_TINY.getbbox(ln); tw=bb[2]-bb[0]
        draw.text((ex1+(EW-tw)//2, lcy), ln, font=F_TINY, fill=col)
        lcy+=18

# arrows from backend to external
arrow(W//2, BY2, W//2, EY1, col=C_EXT)

# ─────────────────────────────────────────────────────────────────────────────
# Legend
# ─────────────────────────────────────────────────────────────────────────────
LX = W - 380; LY = 120
rect(LX, LY, W-40, LY+200, fill="#161B22", outline=BORDER, r=8, lw=1)
text_l(LX+12, LY+8, "Legend", font=F_SMALL, color=C_HEAD)
legend_items = [
    (C_FRONT,  "Frontend (Next.js / TSX)"),
    (C_PIPE,   "Pipeline Stages"),
    (C_DB,     "SQLite Persistence"),
    ("#8957E5", "Target Adapters"),
    (C_EXT,    "External Services"),
    ("#FF7B72", "Azure / DeepEval Judge"),
]
ly2 = LY+34
for col,lbl in legend_items:
    draw.rectangle([LX+14, ly2+3, LX+28, ly2+17], fill=col)
    text_l(LX+36, ly2, lbl, font=F_TINY, color=C_TEXT)
    ly2 += 28

# ─────────────────────────────────────────────────────────────────────────────
OUT = r"c:\Users\Lakshya\Desktop\YASH\metron_final\architecture.png"
img.save(OUT, "PNG", dpi=(150,150))
print("Saved:", OUT)
