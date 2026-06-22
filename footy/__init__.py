"""footy — automated football match-prediction framework.

Two layers coupled by a JSON contract (``params.json``):

* research layer (LLM) -> dossier.md + params.json (team strengths, rho, adjustments)
* numeric layer (deterministic) -> Dixon-Coles score matrix + Monte Carlo -> probabilities

See the project README for the methodology.
"""

__version__ = "0.1.0"
SCHEMA_VERSION = "1.0"
