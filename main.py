"""Food Recommendation System - Main Entry Point.

Provides Flask application factory and CLI commands for:
- Running the API server
- Initializing database constraints
"""

import argparse
import sys

from flask import Flask

from src.config import config
from src.api.routes import api_bp
from src.services.neo4j_service import Neo4jService
from src.pipeline.batch_processor import get_processor


def create_app() -> Flask:
    """Create and configure the Flask application.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__)

    # Configure Flask
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

    # Register blueprints
    app.register_blueprint(api_bp)

    # Add root route
    @app.route("/")
    def index():
        return {
            "name": "Food Recommendation System",
            "version": "0.1.0",
            "description": "API for extracting ingredients from dish images using Gemini AI",
            "endpoints": {
                "health": "/api/v1/health",
                "upload": "POST /api/v1/dishes/upload",
                "process": "POST /api/v1/dishes/process",
                "upload_and_process": "POST /api/v1/dishes/upload-and-process",
                "job_status": "GET /api/v1/jobs/{job_id}/status",
                "get_dish": "GET /api/v1/dishes/{dish_id}",
                "get_ingredients": "GET /api/v1/ingredients",
            },
        }

    return app


def init_database() -> None:
    """Initialize Neo4j database with required constraints."""
    print("Initializing Neo4j database...")

    # Validate config
    missing = config.validate()
    if missing:
        print(f"Error: Missing configuration: {', '.join(missing)}")
        sys.exit(1)

    try:
        neo4j = Neo4jService()
        neo4j.verify_connectivity()
        print("Connected to Neo4j successfully!")

        constraints = neo4j.create_constraints()
        print(f"Created {len(constraints)} constraints:")
        for name in constraints:
            print(f"  - {name}")

        neo4j.close()
        print("Database initialization complete!")

    except Exception as e:
        print(f"Error initializing database: {e}")
        sys.exit(1)


def run_server() -> None:
    """Run the Flask development server."""
    # Validate config
    missing = config.validate()
    if missing:
        print(f"Warning: Missing configuration: {', '.join(missing)}")
        print("Some features may not work correctly.")

    # Ensure temp directory exists
    config.ensure_temp_dir()

    app = create_app()
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Food Recommendation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run the API server
  python main.py --init-db          # Initialize database constraints
  python main.py --host 0.0.0.0     # Run server on specific host
  python main.py --port 8080        # Run server on specific port
        """,
    )

    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize Neo4j database with constraints",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help=f"Host to bind the server (default: {config.FLASK_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Port to bind the server (default: {config.FLASK_PORT})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    args = parser.parse_args()

    if args.init_db:
        init_database()
    else:
        # Override config with CLI args
        if args.host:
            config.FLASK_HOST = args.host
        if args.port:
            config.FLASK_PORT = args.port
        if args.debug:
            config.FLASK_DEBUG = True

        run_server()


if __name__ == "__main__":
    main()
