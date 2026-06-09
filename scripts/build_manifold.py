from affective_manifold.builder import AffectiveManifoldBuilder

builder = AffectiveManifoldBuilder(
    target_vocab_size=4000
)

bundle = builder.build(
    "global_affective_manifold"
)

print("Done!")
print("Vocabulary size:", len(bundle["vocab"]))