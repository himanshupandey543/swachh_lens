from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import config
from .database import init_db
from .routes import analyze, admin_tasks, auth, community, constants, gis, reports

app = FastAPI(title="SwachLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.FRONTEND_ORIGINS + ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# Railway health check
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "SwachLens API"
    }


# API routes
app.include_router(auth.router, prefix="/api")
app.include_router(reports.router, prefix="/api")
app.include_router(constants.router, prefix="/api")
app.include_router(analyze.router, prefix="/api")
app.include_router(gis.router, prefix="/api")
app.include_router(community.router, prefix="/api")
app.include_router(admin_tasks.router, prefix="/api")


@app.on_event("startup")
def _on_startup() -> None:
    init_db()


def _resolve_static(full_path: str) -> Path:
    root = config.STATIC_DIR.resolve()
    candidate = (root / full_path).resolve()

    if candidate != root and root not in candidate.parents:
        raise HTTPException(
            status_code=403,
            detail="Forbidden"
        )

    if candidate.is_dir():
        candidate = candidate / "index.html"

    if candidate.is_file():
        return candidate

    # Extensionless route:
    # /login -> login.html
    if not Path(full_path).suffix:
        html = root / (full_path + ".html")

        if html.is_file():
            return html

    raise HTTPException(
        status_code=404,
        detail="Not found"
    )


@app.get("/{full_path:path}", include_in_schema=False)
def serve_static(full_path: str):
    return FileResponse(
        _resolve_static(full_path),
        headers={
            "Cache-Control": "no-cache, max-age=0"
        },
    )
