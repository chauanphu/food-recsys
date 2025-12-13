"""Gemini API service for extracting ingredients from text descriptions.

Uses Google's Gemini API to analyze dish descriptions and extract
ingredient lists. Image analysis is handled separately by CLIP.

Design Principle:
- Gemini API: Text/description-based ingredient extraction
- CLIP: Image embeddings for visual similarity
"""

import json
import logging
import re
from typing import Any

import google.generativeai as genai

# Configure logger for this module
logger = logging.getLogger(__name__)

from src.config import config


class GeminiExtractor:
    """Service for extracting ingredients from text descriptions using Gemini API.

    This class focuses solely on text-based ingredient extraction.
    For image processing, use the CLIPEmbedder service.
    """

    # Prompt template for ingredient extraction from text descriptions
    TEXT_PROMPT = """Extract all ingredients mentioned in this food dish description.

Description: {description}

Return a JSON object with the following structure:
{{
    "dish_name": "Name of the dish if mentioned, otherwise 'Unknown Dish'",
    "ingredients": ["ingredient1", "ingredient2", ...],
    "cuisine": "Type of cuisine if identifiable, otherwise null",
    "confidence": "high" | "medium" | "low"
}}

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
            self._model = genai.GenerativeModel("gemini-2.5-flash-lite")
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
            text = text.strip()

        # Try to find JSON object if response has extra text
        # Look for pattern starting with { and ending with }
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
        
        logger.debug("Cleaned text for JSON parsing:\n%s", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse Gemini response as JSON: %s", e)
            logger.error("Full raw response:\n%s", text)
            raise ValueError(f"Failed to parse Gemini response as JSON: {e}\nResponse text: {text[:200]}")

    def extract_from_description(
        self,
        description: str,
    ) -> dict[str, Any]:
        """Extract ingredients from a text description.

        This is the primary method for ingredient extraction.
        Gemini analyzes the text description to identify ingredients.

        Args:
            description: Text description of the dish.

        Returns:
            Dictionary with:
                - dish_name: Extracted or inferred dish name
                - ingredients: List of ingredient names
                - cuisine: Type of cuisine if identifiable
                - confidence: Extraction confidence level
                - source: Always "description"

        Raises:
            ValueError: If description is empty or Gemini response cannot be parsed.
        """
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")

        prompt = self.TEXT_PROMPT.format(description=description)
        logger.info("Sending prompt to Gemini for description: %s...", description[:100])
        response = self.model.generate_content(prompt)
        
        # Log raw response for debugging
        logger.info("Gemini raw response text:\n%s", response.text)

        # Parse and return the response
        result = self._parse_json_response(response.text)

        # Validate result is a dictionary
        if not isinstance(result, dict):
            raise ValueError(f"Expected dictionary from Gemini, got {type(result).__name__}")

        # Ensure required fields exist with safe access
        dish_name = result.get("dish_name") if isinstance(result.get("dish_name"), str) else "Unknown Dish"
        ingredients = result.get("ingredients") if isinstance(result.get("ingredients"), list) else []
        cuisine = result.get("cuisine") if isinstance(result.get("cuisine"), str) else None
        confidence = result.get("confidence") if isinstance(result.get("confidence"), str) else "medium"

        return {
            "dish_name": dish_name or "Unknown Dish",
            "ingredients": ingredients,
            "cuisine": cuisine,
            "confidence": confidence,
            "source": "description",
        }

