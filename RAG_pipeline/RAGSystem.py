from RAG_pipeline.Preprocessing_data import DataLoader
from RAG_pipeline.embed import Embedder
from RAG_pipeline.retriever import Retriever
from RAG_pipeline.generator import Generator


class RAGPipeline:
    def __init__(self, data_path, api_key):
        loader = DataLoader(data_path)
        self.df = loader.preprocess()
        self.docs = loader.to_docs(self.df)

        self.products = set(self.df['product'].str.lower().unique())
        self.regions = set(self.df['region'].str.lower().unique())

        self.embedder = Embedder()
        embeddings = self.embedder.encode(self.docs)

        self.retriever = Retriever(embeddings, self.docs)
        self.generator = Generator(api_key)

    def is_aggregation_query(self, query: str) -> bool:
        keywords = ["average", "mean", "sum", "total", "count"]
        return any(k in query.lower() for k in keywords)


    def ask(self, query, k=25):
        q_vec = self.embedder.encode([query])
        context = self.retriever.search(q_vec, k)
        return self.generator.generate(query, context)