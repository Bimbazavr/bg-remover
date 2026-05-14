from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session
from PIL import Image, ImageFilter
import io, os, uuid, base64, subprocess, httpx, numpy as np, sqlite3, urllib.parse, re
from pathlib import Path

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BG_DIR = Path(__file__).parent / "backgrounds"
BG_DIR.mkdir(exist_ok=True)
session = new_session("u2net")
OUT_W, OUT_H = 1080, 1080

DB_PATH = Path(__file__).parent / "prompts.db"
TEMP_IMAGES: dict = {}  # img_id -> bytes (in-memory temp storage)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_MODES = {"color", "bg_name", "bg_file", "generate"}
HEX_RE = re.compile(r'^#[0-9a-fA-F]{6}$')
SAFE_FILENAME_RE = re.compile(r'^[a-zA-Z0-9_\-]+\.(jpg|jpeg|png|webp)$', re.IGNORECASE)

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "img-src 'self' data: blob:; "
        "connect-src 'self';"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


async def read_limited(upload: UploadFile, max_bytes: int = MAX_FILE_SIZE) -> bytes:
    data = await upload.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, "Файл слишком большой (макс. 10 МБ)")
    return data


def safe_bg_path(filename: str) -> Path:
    """Prevent path traversal — only allow safe filenames inside BG_DIR."""
    safe = Path(filename).name
    if not safe or not SAFE_FILENAME_RE.match(safe):
        raise HTTPException(400, "Недопустимое имя файла")
    p = BG_DIR / safe
    try:
        p.resolve().relative_to(BG_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "Недопустимый путь к файлу")
    return p


def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS prompts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        count INTEGER DEFAULT 1,
        score INTEGER DEFAULT 0,
        UNIQUE(text)
    )""")
    con.commit()
    con.close()


def record_prompt(text: str):
    con = sqlite3.connect(DB_PATH)
    con.execute("""INSERT INTO prompts(text,count) VALUES(?,1)
        ON CONFLICT(text) DO UPDATE SET count=count+1""", (text,))
    con.commit()
    con.close()


def augment_prompt(prompt: str) -> str:
    """Enhance prompt using vocabulary from highest-rated historical prompts."""
    try:
        con = sqlite3.connect(DB_PATH)
        rows = con.execute(
            "SELECT text FROM prompts WHERE score >= 0 ORDER BY (count + score * 3) DESC LIMIT 15"
        ).fetchall()
        con.close()
        if not rows:
            return prompt
        prompt_words = set(prompt.lower().split())
        extra_words: list[str] = []
        for (text,) in rows:
            for word in text.lower().split():
                if len(word) > 4 and word not in prompt_words and word not in extra_words:
                    extra_words.append(word)
                if len(extra_words) >= 5:
                    break
            if len(extra_words) >= 5:
                break
        if extra_words:
            return prompt + ", " + ", ".join(extra_words[:3])
    except Exception:
        pass
    return prompt


init_db()


# ── HELPERS ──────────────────────────────────────────────────────────────

def remove_bg(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(remove(data, session=session))).convert("RGBA")


def add_shadow(fg: Image.Image) -> Image.Image:
    canvas = Image.new("RGBA", fg.size, (0, 0, 0, 0))
    blurred = fg.split()[3].filter(ImageFilter.GaussianBlur(22))
    shadow = Image.new("RGBA", fg.size, (15, 15, 15, 115))
    shadow.putalpha(blurred)
    canvas.paste(shadow, (6, 12), shadow)
    canvas.paste(fg, (0, 0), fg)
    return canvas


def compose(fg: Image.Image, bg_img: Image.Image,
            scale: float = 0.82, position: str = "center",
            out_w: int = OUT_W, out_h: int = OUT_H) -> Image.Image:
    bg = bg_img.convert("RGBA").resize((out_w, out_h), Image.LANCZOS)
    fg = fg.copy()
    fg.thumbnail((int(out_w * scale), int(out_h * scale)), Image.LANCZOS)
    x = (out_w - fg.width) // 2
    pad = int(out_h * 0.04)
    y = pad if position == "top" else (out_h - fg.height - pad if position == "bottom"
                                       else (out_h - fg.height) // 2)
    out = Image.new("RGBA", (out_w, out_h))
    out.paste(bg, (0, 0))
    out.paste(fg, (x, y), fg)
    return out.convert("RGB")


def shift_hue(fg: Image.Image, target_hex: str) -> Image.Image:
    target_hex = target_hex.lstrip("#")
    r_t, g_t, b_t = [int(target_hex[i:i+2], 16)/255 for i in (0, 2, 4)]
    mc, mn = max(r_t, g_t, b_t), min(r_t, g_t, b_t)
    d = mc - mn
    if d == 0:
        th = 0.0
    elif mc == r_t:
        th = ((g_t - b_t) / d) % 6 / 6
    elif mc == g_t:
        th = ((b_t - r_t) / d + 2) / 6
    else:
        th = ((r_t - g_t) / d + 4) / 6

    arr = np.array(fg, dtype=np.float32) / 255
    alpha = arr[:, :, 3]
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    diff = maxc - minc
    s = np.where(maxc > 0, diff / maxc, 0.0)
    h = np.zeros_like(r)
    eps = 1e-10
    mr = (maxc == r) & (diff > eps)
    mg = (maxc == g) & (diff > eps) & ~mr
    mb = (maxc == b) & (diff > eps) & ~mr & ~mg
    h[mr] = ((g[mr] - b[mr]) / diff[mr]) % 6 / 6
    h[mg] = ((b[mg] - r[mg]) / diff[mg] + 2) / 6
    h[mb] = ((r[mb] - g[mb]) / diff[mb] + 4) / 6
    h = np.where((s > 0.12) & (alpha > 0.1), th, h)
    h6 = h * 6
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p, q, t2 = v*(1-s), v*(1-f*s), v*(1-(1-f)*s)
    r2 = np.select([i==0,i==1,i==2,i==3,i==4,i==5],[v,q,p,p,t2,v])
    g2 = np.select([i==0,i==1,i==2,i==3,i==4,i==5],[t2,v,v,q,p,p])
    b2 = np.select([i==0,i==1,i==2,i==3,i==4,i==5],[p,p,t2,v,v,q])
    res = arr.copy()
    res[:,:,0], res[:,:,1], res[:,:,2] = r2, g2, b2
    return Image.fromarray((np.clip(res,0,1)*255).astype(np.uint8), "RGBA")


def to_jpeg(img: Image.Image) -> io.BytesIO:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=95)
    buf.seek(0)
    return buf


async def fetch_pollinations(prompt: str, w: int = 1080, h: int = 1080) -> Image.Image:
    seed = uuid.uuid4().int % 99999
    enc = urllib.parse.quote(prompt, safe="")
    url = f"https://image.pollinations.ai/prompt/{enc}?width={w}&height={h}&nologo=true&seed={seed}"
    async with httpx.AsyncClient(timeout=90, follow_redirects=True) as c:
        r = await c.get(url)
        r.raise_for_status()
    return Image.open(io.BytesIO(r.content))


def add_security_headers(response: Response) -> Response:
    for k, v in SECURITY_HEADERS.items():
        response.headers[k] = v
    return response


# ── ROUTES ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    content = open(Path(__file__).parent / "index.html").read()
    response = HTMLResponse(content=content)
    for k, v in SECURITY_HEADERS.items():
        response.headers[k] = v
    return response


@app.post("/api/translate")
async def api_translate(text: str = Form(...)):
    text = text[:500]
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get("https://api.mymemory.translated.net/get",
                            params={"q": text, "langpair": "ru|en"})
            return {"result": r.json()["responseData"]["translatedText"]}
    except Exception:
        return {"result": text}


@app.post("/api/remove")
async def api_remove(file: UploadFile = File(...)):
    data = await read_limited(file)
    fg = remove_bg(data)
    buf = io.BytesIO()
    fg.save(buf, "PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/api/composite")
async def api_composite(
    fg: UploadFile = File(...),
    mode: str = Form("color"),          # color | bg_name | bg_file | generate
    color: str = Form(None),
    bg_name: str = Form(None),
    bg_file: UploadFile = File(None),
    prompt: str = Form(None),
    scale: float = Form(0.82),
    position: str = Form("center"),
    shadow: str = Form("false"),
    hue_color: str = Form(None),        # hex — shift product color
    rotation: float = Form(0),          # degrees
):
    # Validate mode
    if mode not in ALLOWED_MODES:
        raise HTTPException(400, f"Недопустимый режим. Допустимые: {', '.join(ALLOWED_MODES)}")

    # Clamp scale
    scale = max(0.3, min(1.1, scale))

    # Validate color
    if color and not HEX_RE.match(color):
        raise HTTPException(400, "Недопустимый цвет (ожидается #RRGGBB)")

    # Validate hue_color
    if hue_color and hue_color not in ("#000000", "none", "") and not HEX_RE.match(hue_color):
        raise HTTPException(400, "Недопустимый цвет товара (ожидается #RRGGBB)")

    # Truncate prompt
    if prompt:
        prompt = prompt[:500].strip()

    fg_data = await read_limited(fg)
    fg_img = Image.open(io.BytesIO(fg_data)).convert("RGBA")

    # Apply rotation
    if rotation and rotation != 0:
        fg_img = fg_img.rotate(-rotation, expand=True, resample=Image.BICUBIC)

    # Apply hue shift
    if hue_color and hue_color not in ("#000000", "none", ""):
        fg_img = shift_hue(fg_img, hue_color)

    if shadow == "true":
        fg_img = add_shadow(fg_img)

    if mode == "generate":
        if not prompt:
            raise HTTPException(400, "prompt required")
        try:
            bg_img = await fetch_pollinations(prompt)
        except Exception as e:
            raise HTTPException(502, f"Ошибка генерации фона: {e}")
    elif mode == "bg_file" and bg_file and bg_file.filename:
        bg_data = await read_limited(bg_file)
        bg_img = Image.open(io.BytesIO(bg_data))
    elif mode == "bg_name" and bg_name:
        p = safe_bg_path(bg_name)
        if not p.exists():
            raise HTTPException(404, "Фон не найден")
        bg_img = Image.open(p)
    elif mode == "color" and color:
        c = color.lstrip("#")
        rgb = tuple(int(c[i:i+2], 16) for i in (0, 2, 4))
        bg_img = Image.new("RGB", (OUT_W, OUT_H), rgb)
    else:
        bg_img = Image.new("RGB", (OUT_W, OUT_H), (255, 255, 255))

    result = compose(fg_img, bg_img, scale=scale, position=position)
    return StreamingResponse(to_jpeg(result), media_type="image/jpeg")


@app.post("/api/generate-image")
async def api_generate_image(prompt: str = Form(...)):
    prompt = prompt[:500].strip()
    try:
        img = await fetch_pollinations(prompt, 1080, 1080)
        return StreamingResponse(to_jpeg(img), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(502, f"Ошибка генерации: {e}")


def make_reel_frame(img_data: bytes) -> bytes:
    """Place image on 1080×1920 canvas with blurred background (TikTok/Instagram style)."""
    src = Image.open(io.BytesIO(img_data)).convert("RGB")
    # Blurred stretched background
    bg = src.resize((1080, 1920), Image.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(45))
    bg = Image.blend(bg, Image.new("RGB", bg.size, (0, 0, 0)), 0.25)
    # Original centered, max height 1800px
    thumb = src.copy()
    thumb.thumbnail((1080, 1800), Image.LANCZOS)
    x = (1080 - thumb.width) // 2
    y = (1920 - thumb.height) // 2
    bg.paste(thumb, (x, y))
    buf = io.BytesIO()
    bg.save(buf, "JPEG", quality=95)
    buf.seek(0)
    return buf.getvalue()


@app.post("/api/reel")
async def api_reel(image: UploadFile = File(...)):
    data = await read_limited(image)
    frame = make_reel_frame(data)
    tmp_in = f"/tmp/{uuid.uuid4().hex}.jpg"
    tmp_out = f"/tmp/{uuid.uuid4().hex}.mp4"
    Path(tmp_in).write_bytes(frame)
    try:
        cmd = [
            "ffmpeg", "-y", "-loop", "1", "-i", tmp_in,
            "-vf",
            "zoompan=z='if(lte(zoom,1.0),1.06,max(1.001,zoom-0.002))':d=75:"
            "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)',fps=25",
            "-t", "3", "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            tmp_out,
        ]
        res = subprocess.run(cmd, capture_output=True, timeout=60)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.decode()[:300])
        video = Path(tmp_out).read_bytes()
        return StreamingResponse(io.BytesIO(video), media_type="video/mp4",
                                 headers={"Content-Disposition": "attachment; filename=reel.mp4"})
    finally:
        for p in [tmp_in, tmp_out]:
            Path(p).unlink(missing_ok=True)


# ── BACKGROUNDS ──────────────────────────────────────────────────────────

@app.get("/api/backgrounds")
async def list_backgrounds():
    items = []
    for p in sorted(BG_DIR.iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            b64 = base64.b64encode(p.read_bytes()).decode()
            mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
            items.append({"name": p.stem, "file": p.name, "data": f"data:{mime};base64,{b64}"})
    return items


@app.post("/api/backgrounds/upload")
async def upload_background(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        raise HTTPException(400, "Только JPG/PNG/WEBP")
    data = await read_limited(file)
    name = f"{uuid.uuid4().hex[:8]}{ext}"
    (BG_DIR / name).write_bytes(data)
    return {"file": name}


@app.delete("/api/backgrounds/{filename}")
async def delete_background(filename: str):
    p = safe_bg_path(filename)
    if p.exists():
        p.unlink()
    return {"ok": True}


# ── PROMPT LEARNING ──────────────────────────────────────────────────────

@app.get("/api/temp/{img_id}")
async def serve_temp_image(img_id: str):
    # Validate img_id is a hex string (uuid.hex format)
    if not re.match(r'^[0-9a-f]{32}$', img_id):
        raise HTTPException(400, "Invalid image ID")
    data = TEMP_IMAGES.get(img_id)
    if not data:
        raise HTTPException(404, "Temp image not found")
    return StreamingResponse(io.BytesIO(data), media_type="image/jpeg")


@app.post("/api/edit-prompt")
async def api_edit_prompt(request: Request, image: UploadFile = File(...), prompt: str = Form(...)):
    prompt = prompt[:500].strip()
    data = await read_limited(image)
    img_id = uuid.uuid4().hex
    TEMP_IMAGES[img_id] = data

    # Keep memory bounded
    if len(TEMP_IMAGES) > 30:
        del TEMP_IMAGES[next(iter(TEMP_IMAGES))]

    base = str(request.base_url).rstrip("/")
    img_url = f"{base}/api/temp/{img_id}"

    augmented = augment_prompt(prompt)
    enc = urllib.parse.quote(augmented, safe="")
    enc_img = urllib.parse.quote(img_url, safe="")
    seed = uuid.uuid4().int % 99999
    url = (f"https://image.pollinations.ai/prompt/{enc}"
           f"?image={enc_img}&width=1080&height=1080&nologo=true&seed={seed}")

    record_prompt(prompt)

    try:
        async with httpx.AsyncClient(timeout=90, follow_redirects=True) as c:
            r = await c.get(url)
            r.raise_for_status()
        return StreamingResponse(io.BytesIO(r.content), media_type="image/jpeg")
    finally:
        TEMP_IMAGES.pop(img_id, None)


@app.get("/api/prompts/popular")
async def get_popular_prompts(q: str = ""):
    q = q[:200]
    con = sqlite3.connect(DB_PATH)
    if q:
        rows = con.execute(
            "SELECT text, count, score FROM prompts WHERE text LIKE ? ORDER BY (count + score*3) DESC LIMIT 20",
            (f"%{q}%",)
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT text, count, score FROM prompts ORDER BY (count + score*3) DESC LIMIT 20"
        ).fetchall()
    con.close()
    return [{"text": r[0], "count": r[1], "score": r[2]} for r in rows]


@app.post("/api/inpaint")
async def api_inpaint(request: Request, image: UploadFile = File(...),
                      mask: UploadFile = File(...), prompt: str = Form(...)):
    prompt = prompt[:500].strip()
    img_data = await read_limited(image)
    mask_data = await read_limited(mask)

    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    mask_img = Image.open(io.BytesIO(mask_data)).convert("L")

    # Scale mask to match image size
    if mask_img.size != img.size:
        mask_img = mask_img.resize(img.size, Image.LANCZOS)

    bbox = mask_img.getbbox()
    if not bbox:
        raise HTTPException(400, "Маска пустая — обведи область кистью")

    pad = 80
    x1 = max(0, bbox[0] - pad)
    y1 = max(0, bbox[1] - pad)
    x2 = min(img.width, bbox[2] + pad)
    y2 = min(img.height, bbox[3] + pad)

    region = img.crop((x1, y1, x2, y2))
    region_buf = io.BytesIO()
    region.save(region_buf, "JPEG", quality=95)

    img_id = uuid.uuid4().hex
    TEMP_IMAGES[img_id] = region_buf.getvalue()
    if len(TEMP_IMAGES) > 30:
        del TEMP_IMAGES[next(iter(TEMP_IMAGES))]

    base = str(request.base_url).rstrip("/")
    img_url = f"{base}/api/temp/{img_id}"
    enc = urllib.parse.quote(prompt, safe="")
    enc_img = urllib.parse.quote(img_url, safe="")
    seed = uuid.uuid4().int % 99999
    w, h = x2 - x1, y2 - y1
    url = (f"https://image.pollinations.ai/prompt/{enc}"
           f"?image={enc_img}&width={w}&height={h}&nologo=true&seed={seed}")

    record_prompt(prompt)

    try:
        async with httpx.AsyncClient(timeout=90, follow_redirects=True) as c:
            r = await c.get(url)
            r.raise_for_status()

        new_region = Image.open(io.BytesIO(r.content)).convert("RGB").resize((w, h), Image.LANCZOS)
        soft_mask = mask_img.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(12))
        img.paste(new_region, (x1, y1), soft_mask)
        return StreamingResponse(to_jpeg(img), media_type="image/jpeg")
    finally:
        TEMP_IMAGES.pop(img_id, None)


@app.post("/api/prompts/rate")
async def rate_prompt(text: str = Form(...), delta: int = Form(...)):
    text = text[:500]
    # Clamp delta to -1 / 1
    delta = max(-1, min(1, delta))
    con = sqlite3.connect(DB_PATH)
    con.execute("UPDATE prompts SET score=score+? WHERE text=?", (delta, text))
    con.commit()
    con.close()
    return {"ok": True}
