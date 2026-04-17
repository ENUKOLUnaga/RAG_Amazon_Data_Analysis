import pandas as pd

class DataLoader:
    def __init__(self,path:str):
        self.df=pd.read_csv(path)

    def preprocess(self) -> pd.DataFrame:
        df = self.df.drop_duplicates().copy()

        df = df.rename(columns=str.lower) 

        df.fillna({
            "warehouse": "unknown",
            "region": "unknown",
            "product": "unknown",
            "order_qty": 0,
        }, inplace=True)

        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")

        df["delay"] = df["delivery_time_days"].fillna(0)

        df["product"] = df["product"].str.lower()
        df["warehouse"] = df["warehouse"].str.lower()
        df["region"] = df["region"].str.lower()

        return df
    @staticmethod
    def to_docs(df:pd.DataFrame)->list[str]:
        return [
            f"{r.product} from {r.warehouse} {r.region} | "
            f"qty={r.order_qty}, delay={r.delay},status={r.status}"
            for r in df.itertuples()
        ]

