from pathlib import Path
import math
import fitz

OUT = Path(__file__).with_name("Fig1_architecture_redrawn.pdf")
W, H = 1120, 630
S = 519.0 / W  # 183 mm double-column width in points

COL = {
    "ink": (0.09, 0.13, 0.17),
    "blue": (0.84, 0.91, 0.94),
    "existing": (0.91, 0.95, 0.96),
    "green": (0.89, 0.94, 0.90),
    "orange": (0.965, 0.886, 0.784),
    "orange_ink": (0.79, 0.47, 0.13),
    "gray": (0.94, 0.95, 0.96),
    "white": (1, 1, 1),
    "callout": (1.0, 0.99, 0.975),
    "divider": (0.77, 0.80, 0.83),
}

def p(x, y):
    return (x * S, y * S)

def rect(x, y, w, h, fill, stroke=COL["ink"], lw=1.0):
    r = fitz.Rect(x*S, y*S, (x+w)*S, (y+h)*S)
    page.draw_rect(r, color=stroke, fill=fill, width=lw*S)

def line(x1, y1, x2, y2, color=COL["ink"], lw=2.0):
    page.draw_line(p(x1,y1), p(x2,y2), color=color, width=lw*S)

def dashed(x1, y1, x2, y2, color=COL["orange_ink"], lw=2.0):
    dx, dy = x2-x1, y2-y1
    length = math.hypot(dx, dy)
    ux, uy = dx/length, dy/length
    pos = 0.0
    while pos < length:
        a = pos
        b = min(pos+7, length)
        line(x1+ux*a, y1+uy*a, x1+ux*b, y1+uy*b, color, lw)
        pos += 11

def arrow(x1, y1, x2, y2, color=COL["ink"], lw=2.0, dashed_line=False):
    if dashed_line:
        dashed(x1, y1, x2, y2, color, lw)
    else:
        line(x1,y1,x2,y2,color,lw)
    ang = math.atan2(y2-y1, x2-x1)
    ah, aw = 8, 4
    a = (x2-ah*math.cos(ang)+aw*math.sin(ang), y2-ah*math.sin(ang)-aw*math.cos(ang))
    b = (x2-ah*math.cos(ang)-aw*math.sin(ang), y2-ah*math.sin(ang)+aw*math.cos(ang))
    page.draw_polyline([p(*a), p(x2,y2), p(*b)], color=color, fill=color, width=lw*S)

def text(x, y, w, h, value, size=5.6, bold=False, color=COL["ink"]):
    lines = value.split("\n")
    font = "hebo" if bold else "helv"
    size_u = size / S
    line_h = size_u * 1.18
    total_h = line_h * len(lines)
    base_y = (y + max(0, (h - total_h) / 2) + size_u) * S
    for i, line_value in enumerate(lines):
        tw = fitz.get_text_length(line_value, fontname=font, fontsize=size)
        tx = (x + max(0, (w*S - tw) / S) / 2) * S
        page.insert_text((tx, base_y + i*line_h*S), line_value,
                         fontsize=size, fontname=font, color=color)

def multi(x, y, w, lines, size=5.6, bold=False, gap=1.2, color=COL["ink"]):
    h = len(lines) * size * gap + 1
    text(x, y, w, h, "\n".join(lines), size, bold, color)

doc = fitz.open()
page = doc.new_page(width=519, height=630*S)
page.draw_rect(page.rect, color=COL["white"], fill=COL["white"], width=0)

# Title and main boxes
text(350, 12, 420, 22, "RCR-YOLO Architecture", 7.2, True)
rect(28,225,112,86,COL["existing"]); multi(38,245,92,["Input","640 × 640","RGB Image"],5.8,True)
rect(175,104,170,330,COL["existing"]); multi(197,150,126,["YOLO11n","Backbone"],6.3,True); line(198,198,322,198,COL["divider"],1); multi(205,221,110,["CSPDarknet","C3k2","C2PSA"],5.8); multi(205,305,110,["Multi-level feature","extraction"],5.2)
rect(324,120,88,52,COL["blue"]); multi(330,137,76,["P2","160 × 160"],5.4)
rect(324,204,88,52,COL["blue"]); multi(330,221,76,["P3","80 × 80"],5.4)
rect(324,288,88,52,COL["blue"]); multi(330,305,76,["P4","40 × 40"],5.4)
rect(324,372,88,52,COL["blue"]); multi(330,389,76,["P5","20 × 20"],5.4)
rect(324,48,166,54,COL["orange"],COL["orange_ink"],1.0); multi(335,64,145,["ORB-In","Region + Edge Reconstruction"],5.5,True); text(330,30,154,12,"Training only · bypassed at inference",4.9,False,COL["orange_ink"])
rect(450,188,146,190,COL["green"]); multi(470,228,106,["Neck","(PANet)"],6.0,True); multi(475,284,96,["C3k2","C2PSA"],5.6); line(473,327,573,327,COL["divider"],1); multi(468,340,110,["Bottom-up &","top-down fusion"],5.2)

# Three rows
for y, scale, stage in [(174,"P3/16","P3/16"),(278,"P4/32","P4/32"),(382,"P5/64","P5/64")]:
    rect(620,y,122,COL["orange"] if False else 64,COL["orange"],COL["orange_ink"],1.0); multi(628,y+24,106,["MRFE",f"({scale})"],5.8,True)
    rect(780,y,122,64,COL["orange"],COL["orange_ink"],1.0); multi(788,y+24,106,["LCRBase",f"({scale})"],5.8,True)
    rect(938,y,132,64,COL["gray"]); multi(947,y+19,114,["Detect Head",stage,"(Class + Box)"],5.4,True)

# Output and callouts
rect(1070,228,44,178,COL["existing"]); multi(1072,282,40,["Output","boxes","+ classes"],5.0,True)
rect(608,68,142,56,COL["callout"],COL["divider"],0.8); multi(615,81,128,["Multi-Receptive Field","Dilation rates (1, 2, 3)","dynamic weighting"],4.7)
rect(764,68,142,56,COL["callout"],COL["divider"],0.8); multi(771,81,128,["Scene Context","Multi-scale pooling","(1×1, 2×2, 4×4)"],4.7)

# Arrows
arrow(140,268,175,268)
arrow(412,146,450,220,COL["ink"],1.4); arrow(412,230,450,255,COL["ink"],1.4); arrow(412,314,450,290,COL["ink"],1.4); arrow(412,398,450,330,COL["ink"],1.4)
arrow(368,120,397,102,COL["orange_ink"],1.8,True)
arrow(596,224,620,206); arrow(596,283,620,310); arrow(596,342,620,414)
for y in (206,310,414):
    arrow(742,y,780,y); arrow(902,y,938,y)
arrow(1070,206,1070,265); arrow(1070,310,1070,310); arrow(1070,414,1070,369)

# Legend and note
text(30,500,80,14,"Legend",6.2,True)
rect(30,523,18,14,COL["blue"],COL["ink"],0.7); text(56,521,125,18,"Backbone / features",5.0)
rect(205,523,18,14,COL["green"],COL["ink"],0.7); text(231,521,95,18,"PANet neck",5.0)
rect(342,523,18,14,COL["orange"],COL["orange_ink"],0.7); text(368,521,120,18,"Proposed modules",5.0)
dashed(510,530,544,530,COL["orange_ink"],1.5); text(554,521,140,18,"Training-only branch",5.0)
rect(720,505,365,39,COL["callout"],COL["orange_ink"],0.9); multi(735,516,335,["Training: ORB-In provides auxiliary supervision.","Inference: ORB-In is bypassed with zero added cost."],4.9)

doc.save(OUT)
print(OUT)
