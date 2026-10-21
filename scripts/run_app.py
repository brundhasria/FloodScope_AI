"""
FloodScope AI - Application Server Entry Point
Member 3: Application & Integration

Starts the Uvicorn ASGI server hosting the FloodScope AI web application
and candidate change detection API service.

Usage:
    python scripts/run_app.py [--host 127.0.0.1] [--port 8000]
"""

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="FloodScope AI Web Application Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print("=" * 75)
    print("FLOODSCOPE AI -- CANDIDATE CHANGE DETECTION DASHBOARD & API")
    print("=" * 75)
    print(f"Starting server on http://{args.host}:{args.port}")
    print("Open your browser to http://127.0.0.1:8000 to view the interactive dashboard.")
    print("-" * 75)
    print("SCIENTIFIC NOTICE:")
    print("  Detections represent generic candidate surface/radiometric deltas,")
    print("  NOT verified floodwater. Detections are unbenchmarked against")
    print("  independent flood reference labels.")
    print("=" * 75)

    uvicorn.run(
        "floodscope.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
