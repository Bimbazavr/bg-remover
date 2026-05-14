from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
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


def remove_bg(img_bytes: bytes) -> Image.Image:
    result = remove(img_bytes, session=session)
    return Image.open(io.BytesIO(result)).convert("RGBA")


def add_shadow(fg: Image.Image, blur: int = 22, offset: tuple = (6, 12), opacity: float = 0.45) -> Image.Image:
    canvas = Image.new("RGBA", fg.size, (0, 0, 0, 0))
    alpha = fg.split()[3]
    blurred_alpha = alpha.filter(ImageFilter.GaussianBlur(blur))
    shadow_layer = Image.new("RGBA", fg.size, (15, 15, 15, int(255 * opacity)))
    shadow_layer.putalpha(blurred_alpha)
    canvas.paste(shadow_layer, offset, shadow_layer)
    canvas.paste(fg, (0, 0), fg)
    return canvas


def compose(fg: Image.Image, bg_img: Image.Image) -> Image.Image:
    bg = bg_img.convert("RGBA").resize(fg.size, Image.LANCZOS)
    out = Image.new("RGBA", fg.size)
    out.paste(bg, (0, 0))
    out.paste(fg, (0, 0), fg)
    return out.convert("RGB")


def to_jpeg(img: Image.Image, quality: int = 95) -> io.BytesIO:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return buf


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(Path(__file__).parent / "index.html") as f:
        return f.read()


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
    data = await file.read()
    (BG_DIR / name).write_bytes(data)
    return {"file": name}


@app.delete("/api/backgrounds/{filename}")
async def delete_background(filename: str):
    p = BG_DIR / filename
    if p.exists():
        p.unlink()
    return {"ok": True}


@app.post("/api/remove")
async def api_remove(
    file: UploadFile = File(...),
    shadow: str = Form("false"),
):
    data = await file.read()
    fg = remove_bg(data)
    if shadow == "true":
        fg = add_shadow(fg)
    buf = io.BytesIO()
    fg.save(buf, "PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png",
                             headers={"Content-Disposition": "attachment; filename=removed.png"})


@app.post("/api/replace")
async def api_replace(
    file: UploadFile = File(...),
    bg_file: UploadFile = File(None),
    bg_name: str = Form(None),
    color: str = Form(None),
    shadow: str = Form("false"),
):
    data = await file.read()
    fg = remove_bg(data)

    if shadow == "true":
        fg = add_shadow(fg)

    if bg_file and bg_file.filename:
        bg_data = await bg_file.read()
        bg_img = Image.open(io.BytesIO(bg_data))
    elif bg_name:
        p = BG_DIR / bg_name
        if not p.exists():
            raise HTTPException(404, "Фон не найден")
        bg_img = Image.open(p)
    elif color:
        c = color.lstrip("#")
        rgb = tuple(int(c[i:i+2], 16) for i in (0, 2, 4))
        bg_img = Image.new("RGB", fg.size, rgb)
    else:
        raise HTTPException(400, "Укажи фон: bg_file, bg_name или color")

    result = compose(fg, bg_img)
    return StreamingResponse(to_jpeg(result), media_type="image/jpeg",
                             headers={"Content-Disposition": "attachment; filename=result.jpg"})


@app.post("/api/generate")
async def api_generate(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    shadow: str = Form("false"),
):
    data = await file.read()
    fg = remove_bg(data)

    if shadow == "true":
        fg = add_shadow(fg)

    seed = uuid.uuid4().int % 99999
    encoded = prompt.replace(" ", "%20").replace(",", "%2C")
    bg_url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width=1024&height=1024&nologo=true&seed={seed}"
    )

    try:
        async with httpx.AsyncClient(timeout=90, follow_redirects=True) as client:
            resp = await client.get(bg_url)
            resp.raise_for_status()
        bg_img = Image.open(io.BytesIO(resp.content))
    except Exception as e:
        raise HTTPException(502, f"Ошибка генерации фона: {e}")

    result = compose(fg, bg_img)
    return StreamingResponse(to_jpeg(result), media_type="image/jpeg",
                             headers={"Content-Disposition": "attachment; filename=result.jpg"})
