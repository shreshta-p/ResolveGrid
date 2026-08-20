from fastapi import FastAPI

app = FastAPI(title="ResolveGrid API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
