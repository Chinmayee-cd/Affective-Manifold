import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn

import nltk

def ensure_nltk():
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)

ensure_nltk()
class AffectiveProjector:

    def __init__(
        self,
        manifold_path="global_affective_manifold.npz",
        model_name="all-MiniLM-L6-v2"
    ):

        self.model = SentenceTransformer(model_name)

        data = np.load(
            manifold_path,
            allow_pickle=True
        )

        self.vocab = data["vocab"]
        self.definitions = data["definitions"]
        self.word_embeddings = data["word_embeddings"]
        self.manifold_3d = data["manifold_3d"]

        self.pos_anchors = [
            "good",
            "pleasant",
            "joy",
            "love",
            "calm",
            "beautiful"
        ]

        self.neg_anchors = [
            "bad",
            "pain",
            "fear",
            "hate",
            "ugly",
            "anger"
        ]

        self.neu_anchors = [
            "object",
            "thing",
            "entity",
            "item",
            "concept",
            "fact"
        ]

    def _normalize(self, x):

        x = np.asarray(
            x,
            dtype=np.float32
        )

        norms = np.linalg.norm(
            x,
            axis=1,
            keepdims=True
        )

        norms = np.maximum(
            norms,
            1e-12
        )

        return x / norms

    def _encode(self, texts):

        emb = self.model.encode(
            texts,
            show_progress_bar=False
        )

        return self._normalize(emb)

    def project_word(
        self,
        word,
        definition=None
    ):

        if definition is None:

            synsets = wn.synsets(word)

            if not synsets:
                raise ValueError(
                    f"No WordNet entry found for '{word}'"
                )

            definition = synsets[0].definition()

        text = f"{word}: {definition}"

        emb = self._encode([text])[0]

        anchors = self._encode(
            self.pos_anchors +
            self.neg_anchors +
            self.neu_anchors
        )

        p = anchors[:len(self.pos_anchors)]

        n = anchors[
            len(self.pos_anchors):
            len(self.pos_anchors) + len(self.neg_anchors)
        ]

        z = anchors[-len(self.neu_anchors):]

        pos_mean = (emb @ p.T).mean()
        neg_mean = (emb @ n.T).mean()
        neu_mean = (emb @ z.T).mean()

        valence = pos_mean - neg_mean

        # Your newer formula
        arousal = 1.0 - neu_mean

        salience = abs(valence) + arousal

        return {
            "word": word,
            "definition": definition,
            "valence": float(valence),
            "arousal": float(arousal),
            "salience": float(salience)
        }

    def nearest_neighbors(
        self,
        word,
        k=10
    ):

        idx = np.where(
            self.vocab == word
        )[0]

        if len(idx) == 0:
            raise ValueError(
                f"'{word}' not found in manifold."
            )

        idx = idx[0]

        sims = cosine_similarity(
            self.word_embeddings[idx:idx+1],
            self.word_embeddings
        )[0]

        order = sims.argsort()[::-1]

        neighbors = []

        for j in order[1:k+1]:

            neighbors.append(
                (
                    str(self.vocab[j]),
                    float(sims[j])
                )
            )

        return neighbors