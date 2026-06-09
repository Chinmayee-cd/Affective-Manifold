import json
import numpy as np
import nltk

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD

class AffectiveManifoldBuilder:
    def __init__(
        self,
        model_name="all-MiniLM-L6-v2",
        target_vocab_size=4000,
        min_affective_gap=0.08,
        min_salience=0.12,
        random_state=42
    ):
        self.model = SentenceTransformer(model_name)
        self.target_vocab_size = target_vocab_size
        self.min_affective_gap = min_affective_gap
        self.min_salience = min_salience
        self.random_state = random_state
        self.lemmatizer = WordNetLemmatizer()

        self.target_domains = {
            "adj.all", "noun.feeling", "noun.cognition",
            "noun.behavior", "verb.emotion", "verb.social"
        }

        self.pos_anchors = ["good", "pleasant", "joy", "love", "calm", "beautiful"]
        self.neg_anchors = ["bad", "pain", "fear", "hate", "ugly", "anger"]
        self.neu_anchors = ["object", "thing", "entity", "item", "concept", "fact"]

    def _wn_pos(self, syn):
        return {
            "n": "n",
            "v": "v",
            "a": "a",
            "s": "a",
            "r": "r"
        }.get(syn.pos(), "n")

    def _normalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return x / norms

    def _encode(self, texts):
        emb = self.model.encode(texts, show_progress_bar=False)
        return self._normalize(emb)

    def _collect_candidates(self):
        raw_words = []
        raw_defs = []
        seen = set()

        for syn in wn.all_synsets():
            if syn.lexname() not in self.target_domains:
                continue

            lemma = syn.name().split(".")[0]
            if len(lemma) <= 2 or "_" in lemma or not lemma.isalpha():
                continue

            base = self.lemmatizer.lemmatize(lemma.lower(), pos=self._wn_pos(syn))
            if base in seen:
                continue

            seen.add(base)
            raw_words.append(base)
            raw_defs.append(f"{base}: {syn.definition()}")

        return raw_words, raw_defs

    def build(self, output_prefix="affective_manifold"):
        raw_words, raw_defs = self._collect_candidates()
        if not raw_words:
            raise ValueError("No candidates found.")

        word_emb = self._encode(raw_defs)

        anchor_texts = self.pos_anchors + self.neg_anchors + self.neu_anchors
        anchor_emb = self._encode(anchor_texts)

        p = anchor_emb[:len(self.pos_anchors)]
        n = anchor_emb[len(self.pos_anchors):len(self.pos_anchors) + len(self.neg_anchors)]
        z = anchor_emb[-len(self.neu_anchors):]

        pos_score = word_emb @ p.T
        neg_score = word_emb @ n.T
        neu_score = word_emb @ z.T

        pos_mean = pos_score.mean(axis=1)
        neg_mean = neg_score.mean(axis=1)
        neu_mean = neu_score.mean(axis=1)

        valence = pos_mean - neg_mean
        arousal = np.maximum(pos_mean, neg_mean)
        salience = np.abs(valence) + arousal

        keep = (np.abs(valence) >= self.min_affective_gap) & (salience >= self.min_salience)

        vocab = [w for w, k in zip(raw_words, keep) if k]
        defs = [d for d, k in zip(raw_defs, keep) if k]
        emb = word_emb[keep]
        valence = valence[keep]
        arousal = arousal[keep]
        salience = salience[keep]

        if len(vocab) == 0:
            raise ValueError("Filtering was too strict; no words left.")

        if len(vocab) > self.target_vocab_size:
            score = np.abs(valence) + salience
            order = np.argsort(-score)[:self.target_vocab_size]
            vocab = [vocab[i] for i in order]
            defs = [defs[i] for i in order]
            emb = emb[order]
            valence = valence[order]
            arousal = arousal[order]
            salience = salience[order]

        features = np.column_stack([valence, arousal, salience]).astype(np.float32)

        svd_dim = min(32, emb.shape[0] - 1, emb.shape[1])
        if svd_dim >= 2:
            svd = TruncatedSVD(n_components=svd_dim, random_state=self.random_state)
            reduced = svd.fit_transform(emb)
        else:
            reduced = emb.astype(np.float32)

        bundle = {
            "vocab": vocab,
            "definitions": defs,
            "word_embeddings": emb.astype(np.float32),
            "manifold_3d": features,
            "reduced_embeddings": reduced.astype(np.float32),
            "anchors": {
                "positive": self.pos_anchors,
                "negative": self.neg_anchors,
                "neutral": self.neu_anchors
            },
            "config": {
                "target_vocab_size": self.target_vocab_size,
                "min_affective_gap": self.min_affective_gap,
                "min_salience": self.min_salience,
                "model_name": self.model._first_module().__class__.__name__ if hasattr(self.model, "_first_module") else "SentenceTransformer"
            }
        }

        np.savez_compressed(
            f"{output_prefix}.npz",
            vocab=np.array(vocab, dtype=object),
            definitions=np.array(defs, dtype=object),
            word_embeddings=emb.astype(np.float32),
            manifold_3d=features,
            reduced_embeddings=reduced.astype(np.float32)
        )

        with open(f"{output_prefix}.json", "w", encoding="utf-8") as f:
            json.dump(bundle["config"] | {"anchors": bundle["anchors"]}, f, indent=2)

        self.vocab = vocab
        self.definitions = defs
        self.word_embeddings = emb
        self.manifold_3d = features
        self.reduced_embeddings = reduced
        self.bundle = bundle

        return bundle

    def project_word(self, word, definition=None):
        if definition is None:
            synsets = wn.synsets(word)
            if not synsets:
                raise ValueError(f"No WordNet synsets found for '{word}'.")
            definition = synsets[0].definition()

        text = f"{word}: {definition}"
        emb = self._encode([text])[0]

        anchors = self._encode(self.pos_anchors + self.neg_anchors + self.neu_anchors)
        p = anchors[:len(self.pos_anchors)]
        n = anchors[len(self.pos_anchors):len(self.pos_anchors) + len(self.neg_anchors)]
        z = anchors[-len(self.neu_anchors):]

        pos_mean = (emb @ p.T).mean()
        neg_mean = (emb @ n.T).mean()
        neu_mean = (emb @ z.T).mean()

        valence = pos_mean - neg_mean
        arousal = np.maximum(pos_mean, neg_mean) - neu_mean
        salience = np.abs(valence) + arousal

        return {
            "word": word,
            "definition": definition,
            "valence": float(valence),
            "arousal": float(arousal),
            "salience": float(salience)
        }