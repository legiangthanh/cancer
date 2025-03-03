class InMemoryVectorDB:
    def __init__(self):
        self.collections = {}

    def get_or_create_collection(self, name):
        if name not in self.collections:
            self.collections[name] = Collection()
        return self.collections[name]


def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    if magnitude1 * magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)


class Collection:
    def __init__(self):
        self.data = []

    def add(self, data):
        self.data.extend(data)

    def query(self, query, num):
        similarities = [cosine_similarity(query, embedding) for embedding in self.data]
        best_match_index = sorted(range(len(similarities)), key=lambda sub: similarities[sub])[-num:]
        print(best_match_index)
        return [self.data[index] for index in best_match_index]
