import sys
import argparse
import uvicorn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))         # src/ - for agent imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # project root - for api imports

from agent.trainer import clear_model
from agent.game import clear_experiences


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror AI Chess Bot")
    parser.add_argument(
        "--reset-model",
        action="store_true",
        help="Delete saved model weights and exit.",
    )
    parser.add_argument(
        "--reset-data",
        action="store_true",
        help="Delete all saved game experience and exit.",
    )
    parser.add_argument(
        "--reset-all",
        action="store_true",
        help="Delete both model weights and game experience, then exit.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000).",
    )
    args = parser.parse_args()

    if args.reset_all:
        clear_model()
        clear_experiences()
        return

    if args.reset_model:
        clear_model()
        return

    if args.reset_data:
        clear_experiences()
        return

    print(f"Starting server at http://{args.host}:{args.port}")
    print("Open that URL in your browser to play.\n")
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
