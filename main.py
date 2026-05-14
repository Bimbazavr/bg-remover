from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session
from PIL import Image, ImageFilter
import io, os, uuid, base64, subprocess, httpx, numpy as np
from pathlib import Path

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BG_DIR = Path(__file__).parent / "backgrounds"
BG_DIR.mkdir(exist_ok=True)
session = new_session("u2net")
OUT_W, OUT_H = 1080, 1080


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
    enc = prompt.replace(" ", "%20").replace(",", "%2C")
    url = f"https://image.pollinations.ai/prompt/{enc}?width={w}&height={h}&nologo=true&seed={seed}"
    async with httpx.AsyncClient(timeout=90, follow_redirects=True) as c:
        r = await c.get(url)
        r.raise_for_status()
    return Image.open(io.BytesIO(r.content))


# ── ROUTES ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return open(Path(__file__).parent / "index.html").read()


@app.post("/api/translate")
async def api_translate(text: str = Form(...)):
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get("https://api.mymemory.translated.net/get",
                            params={"q": text, "langpair": "ru|en"})
            return {"result": r.json()["responseData"]["translatedText"]}
    except Exception:
        return {"result": text}


@app.post("/api/remove")
async def api_remove(file: UploadFile = File(...)):
    fg = remove_bg(await file.read())
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
    fg_img = Image.open(io.BytesIO(await fg.read())).convert("RGBA")

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
        bg_img = Image.open(io.BytesIO(await bg_file.read()))
    elif mode == "bg_name" and bg_name:
        p = BG_DIR / bg_name
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
    try:
        img = await fetch_pollinations(prompt, 1080, 1080)
        return StreamingResponse(to_jpeg(img), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(502, f"Ошибка генерации: {e}")


@app.post("/api/reel")
async def api_reel(image: UploadFile = File(...)):
    data = await image.read()
    tmp_in = f"/tmp/{uuid.uuid4().hex}.jpg"
    tmp_out = f"/tmp/{uuid.uuid4().hex}.mp4"
    Path(tmp_in).write_bytes(data)
    try:
        cmd = [
            "ffmpeg", "-y", "-loop", "1", "-i", tmp_in,
            "-vf",
            "scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,"
            "zoompan=z='min(zoom+0.002,1.5)':d=75:"
            "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)',fps=25",
            "-t", "3", "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
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
    name = f"{uuid.uuid4().hex[:8]}{ext}"
    (BG_DIR / name).write_bytes(await file.read())
    return {"file": name}


@app.delete("/api/backgrounds/{filename}")
async def delete_background(filename: str):
    p = BG_DIR / filename
    if p.exists():
        p.unlink()
    return {"ok": True}
