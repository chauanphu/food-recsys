---
applyTo: '**'
---
# Food Recomemndation System Instructions

## Dish Similarity Calculation
To calculate the similarity between dishes based on their ingredients, use the Jaccard similarity coefficient. The Jaccard similarity between two sets A and B is defined as the size of the intersection divided by the size of the union of the sets:
```
J(A, B) = |Ingredients(A) ∩ Ingredients(B)| / |Ingredients(A) ∪ Ingredients(B)|
```
This metric will help identify dishes that share similar ingredients, which can be useful for content-based filtering in the recommendation engine.

Or, we can can compute the dish vector as the average of its ingredient embeddings. Then, use cosine similarity to measure the similarity between dish vectors.

Or, we can also compute the dish similarity based on the cosine on their image embeddings. This will recommend visually similar dishes.

## Dietary Restrictions Handling
When filtering dishes based on dietary restrictions, ensure that the ingredients of each dish do not contain any items from the user's list of restricted ingredients. For example, if a user is allergic to nuts, exclude any dish that contains nuts from the recommendation list.
