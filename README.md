# Affective Manifold

Affective Manifold is a lightweight Python library for projecting words into an interpretable affective space using transformer embeddings and WordNet-derived emotional concepts.

The library computes three affective dimensions:

- **Valence** — positive vs. negative emotional orientation
- **Arousal** — emotional intensity
- **Salience** — overall affective significance

A precomputed manifold containing 664 affective concepts is included with the package, enabling immediate use without additional preprocessing.

## Installation

```bash
pip install affective-manifold
```

## Usage

```python
from affective_manifold import AffectiveProjector

proj = AffectiveProjector()

print(proj.project_word("joy"))
```

Example output:

```python
{
    'word': 'joy',
    'definition': 'the emotion of great happiness',
    'valence': 0.1166,
    'arousal': 0.8477,
    'salience': 0.9643
}
```

Find semantically related affective concepts:

```python
from affective_manifold import AffectiveProjector

proj = AffectiveProjector()

print(proj.nearest_neighbors("anxiety"))
```

Example output:

```python
[
    ('discomfort', 0.72),
    ('fear', 0.70),
    ('insecurity', 0.64),
    ('apprehensive', 0.59)
]
```
