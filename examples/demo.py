from affective_manifold import AffectiveProjector

proj = AffectiveProjector(
    manifold_path="global_affective_manifold.npz"
)

print(
    proj.project_word("melancholy")
)

print(
    proj.nearest_neighbors("fear", k=10)
)