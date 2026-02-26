import pickle

import numpy as np
from sklearn import cluster


def main():
    print("Start...")
    kmeans = cluster.MiniBatchKMeans(
        n_clusters=1024,
        verbose=1,
        batch_size=10000,
        max_iter=100,
        max_no_improvement=50
        # tol=1e-5,
    )

    file_path_embedding_metadata = "./final_embed/embedding_metadata"
    with open(file_path_embedding_metadata, "rb") as f:
        embedding_metadata = pickle.load(f)

    file_path_embeddings = "./final_embed/embeddings.npy"
    with open(file_path_embeddings, "rb") as f:
        embeddings = np.reshape(
            np.fromfile(f, dtype=np.float32),
            shape=(
                embedding_metadata["num_docs"],
                embedding_metadata["truncate_dim"],
            ),
        )

    clusters_docs = kmeans.fit_transform(embeddings)
    print(kmeans.cluster_centers_)
    print(clusters_docs)

    with open("./final_embed/kmeans.pkl", "wb") as f:
        pickle.dump(kmeans, f)


if __name__ == "__main__":
    main()
