"""Gemini API service for extracting ingredients from images and text.

Uses Google's Gemini multimodal API to analyze dish images and
descriptions to extract ingredient lists.
"""

import json
import re
from pathlib import Path
from typing import Any

import google.generativeai as genai
from PIL import Image

from src.config import config


class GeminiExtractor:
    """Service for extracting ingredients using Gemini API."""

    # Prompt template for ingredient extraction from images
    IMAGE_PROMPT = """Analyze this food dish image and extract all visible or likely ingredients.

Return a JSON object with the following structure:
{
    "dish_name": "Name of the dish if identifiable, otherwise 'Unknown Dish'",
    "ingredients": ["ingredient1", "ingredient2", ...],
    "cuisine": "Type of cuisine if identifiable (e.g., 'Italian', 'Japanese'), otherwise null",
    "confidence": "high" | "medium" | "low"
}

Guidelines:
- List all ingredients you can identify or reasonably infer from the dish
- Use lowercase for ingredient names
- Be specific (e.g., "olive oil" not just "oil", "parmesan cheese" not just "cheese")
- Include seasonings and garnishes if visible
- If you cannot identify any ingredients, return an empty list

Respond with ONLY the JSON object, no additional text."""

    # Prompt template for ingredient extraction from text descriptions
    TEXT_PROMPT = """Extract all ingredients mentioned in this food dish description.

Description: {description}

Return a JSON object with the following structure:
{
    "dish_name": "Name of the dish if mentioned, otherwise 'Unknown Dish'",
    "ingredients": ["ingredient1", "ingredient2", ...],
    "cuisine": "Type of cuisine if identifiable, otherwise null",
    "confidence": "high" | "medium" | "low"
}

Guidelines:
- Extract all explicitly mentioned ingredients
- Infer common ingredients if the dish type is mentioned (e.g., "pizza" implies dough, tomato sauce, cheese)
- Use lowercase for ingredient names
- Be specific with ingredient names

Respond with ONLY the JSON object, no additional text."""

    def __init__(self, api_key: str | None = None):
        """Initialize the Gemini extractor.

        Args:
            api_key: Gemini API key. Defaults to config value.
        """
        self._api_key = api_key or config.GEMINI_API_KEY
        self._model: genai.GenerativeModel | None = None
        self._configure()

    def _configure(self) -> None:
        """Configure the Gemini API client."""
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY is required")
        genai.configure(api_key=self._api_key)

    @property
    def model(self) -> genai.GenerativeModel:
        """Get or create the Gemini model instance."""
        if self._model is None:
            self._model = genai.GenerativeModel("gemini-1.5-flash")
        return self._model

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Parse JSON from Gemini response, handling markdown code blocks.

        Args:
            text: Raw response text from Gemini.

        Returns:
            Parsed JSON dictionary.

        Raises:
            ValueError: If JSON parsing fails.
        """
        # Remove markdown code blocks if present
        text = text.strip()
        if text.startswith("```"):
            # Remove ```json or ``` at start and ``` at end
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Gemini response as JSON: {e}")

    def extract_from_image(
        self,
        image_path: str | Path,
    ) -> dict[str, Any]:
        """Extract ingredients from a dish image.

        Args:
            image_path: Path to the image file.

        Returns:
            Dictionary with dish_name, ingredients, cuisine, and confidence.

        Raises:
            FileNotFoundError: If image file doesn't exist.
            ValueError: If Gemini response cannot be parsed.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load and prepare image
        image = Image.open(image_path)

        # Generate content with image and prompt
        response = self.model.generate_content([self.IMAGE_PROMPT, image])

        # Parse and return the response
        result = self._parse_json_response(response.text)

        # Ensure required fields exist
        return {
            "dish_name": result.get("dish_name", "Unknown Dish"),
            "ingredients": result.get("ingredients", []),
            "cuisine": result.get("cuisine"),
            "confidence": result.get("confidence", "medium"),
            "source": "image",
            "image_path": str(image_path),
        }

    def extract_from_description(
        self,
        description: str,
    ) -> dict[str, Any]:
        """Extract ingredients from a text description.

        Args:
            description: Text description of the dish.

        Returns:
            Dictionary with dish_name, ingredients, cuisine, and confidence.

        Raises:
            ValueError: If Gemini response cannot be parsed.
        """
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")

        prompt = self.TEXT_PROMPT.format(description=description)
        response = self.model.generate_content(prompt)

        # Parse and return the response
        result = self._parse_json_response(response.text)

        # Ensure required fields exist
        return {
            "dish_name": result.get("dish_name", "Unknown Dish"),
            "ingredients": result.get("ingredients", []),
            "cuisine": result.get("cuisine"),
            "confidence": result.get("confidence", "medium"),
            "source": "description",
            "description": description,
        }

    def extract_from_image_and_description(
        self,
        image_path: str | Path,
        description: str,
    ) -> dict[str, Any]:
        """Extract ingredients from both image and description.

        Combines results from both sources for better accuracy.

        Args:
            image_path: Path to the image file.
            description: Text description of the dish.

        Returns:
            Dictionary with combined dish_name, ingredients, cuisine, and confidence.
        """
        image_path = Path(image_path)

        # Load image
        image = Image.open(image_path)

        # Combined prompt
        combined_prompt = f"""Analyze this food dish image along with its description and extract all ingredients.

Description: {description}

Return a JSON object with the following structure:
{{
    "dish_name": "Name of the dish",
    "ingredients": ["ingredient1", "ingredient2", ...],
    "cuisine": "Type of cuisine if identifiable, otherwise null",
    "confidence": "high" | "medium" | "low"
}}

Guidelines:
- Combine information from both the image and description
- List all ingredients visible in the image or mentioned in the description
- Use lowercase for ingredient names
- Be specific with ingredient names
- Resolve any conflicts by preferring the more specific information

Respond with ONLY the JSON object, no additional text."""

        response = self.model.generate_content([combined_prompt, image])
        result = self._parse_json_response(response.text)

        return {
            "dish_name": result.get("dish_name", "Unknown Dish"),
            "ingredients": result.get("ingredients", []),
            "cuisine": result.get("cuisine"),
            "confidence": result.get("confidence", "medium"),
            "source": "image_and_description",
            "image_path": str(image_path),
            "description": description,
        }
