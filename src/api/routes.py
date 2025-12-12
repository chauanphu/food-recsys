"""Flask API routes for dish image upload and ingredient extraction.

Provides endpoints for:
- Batch image upload
- Processing uploaded images with Gemini
- Checking job status
"""

import os
import tempfile
import uuid
from pathlib import Path

from flask import Blueprint, request, jsonify, Response
from werkzeug.utils import secure_filename

from src.config import config
from src.pipeline.batch_processor import get_processor, ProcessingStatus

# Create API blueprint
api_bp = Blueprint("api", __name__, url_prefix="/api/v1")


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed.

    Args:
        filename: The filename to check.

    Returns:
        True if extension is allowed.
    """
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in config.ALLOWED_EXTENSIONS


def save_uploaded_file(file) -> dict:
    """Save an uploaded file to temporary storage.

    Args:
        file: Werkzeug FileStorage object.

    Returns:
        Dictionary with file metadata.
    """
    filename = secure_filename(file.filename)
    file_id = str(uuid.uuid4())
    ext = Path(filename).suffix

    # Ensure temp directory exists
    config.ensure_temp_dir()

    # Create temp file
    temp_path = config.TEMP_UPLOAD_DIR / f"{file_id}{ext}"
    file.save(temp_path)

    return {
        "id": file_id,
        "original_name": filename,
        "temp_path": str(temp_path),
    }


@api_bp.route("/health", methods=["GET"])
def health_check() -> Response:
    """Health check endpoint.

    Returns:
        JSON response with status.
    """
    return jsonify({"status": "healthy", "service": "food-recsys"})


@api_bp.route("/dishes/upload", methods=["POST"])
def upload_dish_images() -> tuple[Response, int]:
    """Upload dish images for ingredient extraction.

    Accepts multipart/form-data with:
    - images: One or more image files
    - descriptions: Optional JSON array of descriptions (matching image order)

    Returns:
        JSON response with upload results and item IDs for processing.
    """
    if "images" not in request.files:
        return jsonify({"error": "No images provided", "code": "NO_IMAGES"}), 400

    files = request.files.getlist("images")

    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files selected", "code": "EMPTY_FILES"}), 400

    # Get optional descriptions
    descriptions = request.form.getlist("descriptions")

    uploaded = []
    errors = []

    for idx, file in enumerate(files):
        if not file or file.filename == "":
            continue

        if allowed_file(file.filename):
            try:
                item = save_uploaded_file(file)

                # Add description if provided
                if idx < len(descriptions):
                    item["description"] = descriptions[idx]

                uploaded.append(item)
            except Exception as e:
                errors.append({
                    "filename": file.filename,
                    "error": str(e),
                })
        else:
            errors.append({
                "filename": file.filename,
                "error": f"Invalid file type. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}",
            })

    if not uploaded:
        return jsonify({
            "error": "No valid files uploaded",
            "errors": errors,
            "code": "NO_VALID_FILES",
        }), 400

    return jsonify({
        "uploaded": uploaded,
        "errors": errors,
        "total": len(uploaded),
        "failed": len(errors),
    }), 200


@api_bp.route("/dishes/process", methods=["POST"])
def process_dishes() -> tuple[Response, int]:
    """Start processing uploaded dish images.

    Accepts JSON body with:
    - items: Array of items from upload response (with id, temp_path, optional description)

    Returns:
        JSON response with job_id for tracking.
    """
    data = request.get_json()

    if not data or "items" not in data:
        return jsonify({
            "error": "No items provided",
            "code": "NO_ITEMS",
        }), 400

    items = data["items"]

    if not items:
        return jsonify({
            "error": "Items array is empty",
            "code": "EMPTY_ITEMS",
        }), 400

    # Validate items
    valid_items = []
    for item in items:
        if "id" not in item or "temp_path" not in item:
            continue

        # Check if file exists
        if not os.path.exists(item["temp_path"]):
            continue

        valid_items.append(item)

    if not valid_items:
        return jsonify({
            "error": "No valid items to process",
            "code": "NO_VALID_ITEMS",
        }), 400

    # Start batch processing
    processor = get_processor()
    job_id = processor.start_batch(valid_items)

    return jsonify({
        "job_id": job_id,
        "status": "processing",
        "total_items": len(valid_items),
        "status_url": f"/api/v1/jobs/{job_id}/status",
    }), 202


@api_bp.route("/dishes/upload-and-process", methods=["POST"])
def upload_and_process() -> tuple[Response, int]:
    """Upload and immediately process dish images.

    Convenience endpoint that combines upload and process steps.

    Accepts multipart/form-data with:
    - images: One or more image files
    - descriptions: Optional descriptions (matching image order)

    Returns:
        JSON response with job_id for tracking.
    """
    if "images" not in request.files:
        return jsonify({"error": "No images provided", "code": "NO_IMAGES"}), 400

    files = request.files.getlist("images")

    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files selected", "code": "EMPTY_FILES"}), 400

    descriptions = request.form.getlist("descriptions")

    items = []
    errors = []

    for idx, file in enumerate(files):
        if not file or file.filename == "":
            continue

        if allowed_file(file.filename):
            try:
                item = save_uploaded_file(file)

                if idx < len(descriptions):
                    item["description"] = descriptions[idx]

                items.append(item)
            except Exception as e:
                errors.append({
                    "filename": file.filename,
                    "error": str(e),
                })
        else:
            errors.append({
                "filename": file.filename,
                "error": f"Invalid file type. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}",
            })

    if not items:
        return jsonify({
            "error": "No valid files to process",
            "errors": errors,
            "code": "NO_VALID_FILES",
        }), 400

    # Start batch processing
    processor = get_processor()
    job_id = processor.start_batch(items)

    return jsonify({
        "job_id": job_id,
        "status": "processing",
        "total_items": len(items),
        "upload_errors": errors,
        "status_url": f"/api/v1/jobs/{job_id}/status",
    }), 202


@api_bp.route("/jobs/<job_id>/status", methods=["GET"])
def get_job_status(job_id: str) -> tuple[Response, int]:
    """Get the status of a processing job.

    Args:
        job_id: The job identifier.

    Returns:
        JSON response with job status and results.
    """
    processor = get_processor()
    job = processor.get_job(job_id)

    if job is None:
        return jsonify({
            "error": "Job not found",
            "code": "JOB_NOT_FOUND",
        }), 404

    return jsonify(job.to_dict()), 200


@api_bp.route("/dishes/<dish_id>", methods=["GET"])
def get_dish(dish_id: str) -> tuple[Response, int]:
    """Get a dish by ID from the database.

    Args:
        dish_id: The dish identifier.

    Returns:
        JSON response with dish data.
    """
    processor = get_processor()
    dish = processor.neo4j.get_dish_by_id(dish_id)

    if dish is None:
        return jsonify({
            "error": "Dish not found",
            "code": "DISH_NOT_FOUND",
        }), 404

    return jsonify(dish), 200


@api_bp.route("/ingredients", methods=["GET"])
def get_all_ingredients() -> tuple[Response, int]:
    """Get all ingredients from the database.

    Returns:
        JSON response with list of ingredient names.
    """
    processor = get_processor()
    ingredients = processor.neo4j.get_all_ingredients()

    return jsonify({
        "ingredients": ingredients,
        "count": len(ingredients),
    }), 200
