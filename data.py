import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import reuters

nltk.download('reuters')
nltk.download('punkt')


documents = [reuters.raw(doc_id) for doc_id in reuters.fileids()]


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)


clusters = kmeans.labels_.tolist()


for i in range(num_clusters):
    print(f"\nCluster {i}:")
    cluster_docs = [documents[j] for j in range(len(clusters)) if clusters[j] == i]
    for doc in cluster_docs[:5]:  
        print(doc[:200])  


with open("saida_clusters.txt", "w") as f:
    for i in range(num_clusters):
        f.write(f"\nCluster {i}:\n")
        cluster_docs = [documents[j] for j in range(len(clusters)) if clusters[j] == i]
        for doc in cluster_docs[:5]:  
            f.write(doc[:200] + "\n")  


# Se seus clusters vierem como strings grandes, você pode organizá-los assim:
cluster_0 = """
ECONOMIC SPOTLIGHT - AUSTRALIAN MARKETS BOOMING
BANK OF JAPAN INTERVENES SOON AFTER TOKYO OPENING
POEHL WARNS AGAINST FURTHER DOLLAR FALL
"""

cluster_1 = """
AUSTRALIAN FOREIGN SHIP BAN ENDS BUT NSW PORTS HIT
WESTERN MINING TO OPEN NEW GOLD MINE IN AUSTRALIA
"""

# Transformando as strings em listas
clusters = [cluster_0.strip().split('\n'), cluster_1.strip().split('\n')]
