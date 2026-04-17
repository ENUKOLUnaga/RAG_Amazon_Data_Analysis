
from RAG_pipeline.RAGSystem import RAGPipeline

def main():
    file=f"E:/RAG_AWS_Supply_Chain/data/aws_supply_chain_orders_raw (1).csv"

    api="YOUR_GROQ_API_KEY"
    rag=RAGPipeline(file,api_key=api)

    while True:
        q=input("\nAsk: ")
        if q.lower()=="exit":
            break

        print("\n",rag.ask(q))

if __name__=="__main__":
    main()
