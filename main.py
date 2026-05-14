from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session
from PIL import Image, ImageFilter
import io
import os
import uuid
import base64
import httpx
from pathlib import Path

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BG_DIR = Path(__file__).parent / "backgrounds"
BG_DIR.mkdir(exist_ok=True)

session = new_session("u2net")

OUT_SIZE = (1024, 1024)


def remove_bg(img_bytes: bytes) -> Image.Image:
    result = remove(img_bytes, session=session)
    return Image.open(io.BytesIO(result)).convert("RGBA")


def add_shadow(fg: Image.Image, blur: int = 22, offset: tuple = (6, 12), opacity: float = 0.45) -> Image.Image:
    canvas = Image.new("RGBA", fg.size, (0, 0, 0, 0))
    blurred = fg.split()[3].filter(ImageFilter.GaussianBlur(blur))
    shadow = Image.new("RGBA", fg.size, (15, 15, 15, int(255 * opacity)))
    shadow.putalpha(blurred)
    canvas.paste(shadow, offset, shadow)
    canvas.paste(fg, (0, 0), fg)
    return canvas


def compose(fg: Image.Image, bg_img: Image.Image, scale: float = 0.82, position: str = "center") -> Image.Image:
    bg = bg_img.convert("RGBA").resize(OUT_SIZE, Image.LANCZOS)
    fg = fg.copy()

    max_w = int(OUT_SIZE[0] * scale)
    max_h = int(OUT_SIZE[1] * scale)
    fg.thumbnail((max_w, max_h), Image.LANCZOS)

    x = (OUT_SIZE[0] - fg.width) // 2
    pad = int(OUT_SIZE[1] * 0.04)
    if position == "top":
        y = pad
    elif position == "bottom":
        y = OUT_SIZE[1] - fg.height - pad
    else:
        y = (OUT_SIZE[1] - fg.height) // 2

    out = Image.new("RGBA", OUT_SIZE)
    out.paste(bg, (0, 0))
    out.paste(fg, (x, y), fg)
    return out.convert("RGB")


def to_jpeg(img: Image.Image) -> io.BytesIO:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=95)
    buf.seek(0)
    return buf


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(Path(__file__).parent / "index.html") as f:
        return f.read()


# ── Translate RU→EN via MyMemory (free, no key) ────────────────────────────
@app.post("/api/translate")
async def api_translate(text: str = Form(...)):
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(
                "https://api.mymemory.translated.net/get",
                params={"q": text, "langpair": "ru|en"},
            )
            data = r.json()
            return {"result": data["responseData"]["translatedText"]}
    except Exception:
        return {"result": text}  # fallback: send as-is


# ── Remove background ───────────────────────────────────────────────────────
@app.post("/api/remove")
async def api_remove(file: UploadFile = File(...)):
    fg = remove_bg(await file.read())
    buf = io.BytesIO()
    fg.save(buf, "PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# ── Composite: fg PNG + background + scale/position ────────────────────────
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
):
    fg_img = Image.open(io.BytesIO(await fg.read())).convert("RGBA")

    if shadow == "true":
        fg_img = add_shadow(fg_img)

    if mode == "generate":
        if not prompt:
            raise HTTPException(400, "prompt required")
        seed = uuid.uuid4().int % 99999
        encoded = prompt.replace(" ", "%20").replace(",", "%2C")
        bg_url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true&seed={seed}"
        try:
            async with httpx.AsyncClient(timeout=90, follow_redirects=True) as client:
                resp = await client.get(bg_url)
                resp.raise_for_status()
            bg_img = Image.open(io.BytesIO(resp.content))
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
        bg_img = Image.new("RGB", OUT_SIZE, rgb)

    else:
        raise HTTPException(400, "Укажи режим фона")

    result = compose(fg_img, bg_img, scale=scale, position=position)
    return StreamingResponse(to_jpeg(result), media_type="image/jpeg")


# ── Backgrounds library ─────────────────────────────────────────────────────
@app.get("/api/backgrounds")
async def list_backgrounds():
    items = []
    for p in sorted(BG_DIR.iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
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
