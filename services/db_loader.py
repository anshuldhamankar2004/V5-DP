import polars as pl
import pandas as pd  # Required for most connectors
from sqlalchemy import create_engine
from pymongo import MongoClient
import pyarrow as pa
import google
def load_from_database(db_type, host, port, user, password, database, table):
    if db_type == "sql":
        if port == "5432":
            dialect = "postgresql"
        elif port == "3306":
            dialect = "mysql+pymysql"
        elif port == "1433":
            dialect = "mssql+pymssql"
        else:
            raise ValueError(f"Unsupported port {port} for SQL connection")

        engine = create_engine(f"{dialect}://{user}:{password}@{host}:{port}/{database}")

        # Clean table name if passed like "sales/products"
        if "/" in table:
            schema, table_name = table.split("/", 1)
            full_table_name = f"{schema}.{table_name}"
        else:
            full_table_name = table

        df = pd.read_sql(f"SELECT * FROM {full_table_name}", engine)

    elif db_type == "nosql":
        client = MongoClient(f"mongodb://{user}:{password}@{host}:{port}/")
        db = client[database]
        collection = db[table]
        data = list(collection.find())
        df = pd.DataFrame(data)
        return pl.from_pandas(df)

    elif db_type == "aws":
        # Example: Redshift
        engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}")
        df = pd.read_sql(f"SELECT * FROM {table}", engine)
        return pl.from_pandas(df)

    elif db_type == "gcp":
        from google.cloud import bigquery
        client = bigquery.Client()
        query = f"SELECT * FROM `{database}.{table}`"
        df = client.query(query).to_dataframe()
        return pl.from_pandas(df)

    elif db_type == "snowflake":
        import snowflake.connector
        conn = snowflake.connector.connect(
            user=user,
            password=password,
            account=host,
            database=database
        )
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {table}")
        df = pd.DataFrame(cur.fetchall(), columns=[col[0] for col in cur.description])
        return pl.from_pandas(df)

    elif db_type == "azure":
        engine = create_engine(f"mssql+pymssql://{user}:{password}@{host}:{port}/{database}")
        df = pd.read_sql(f"SELECT * FROM {table}", engine)
        return pl.from_pandas(df)

    elif db_type == "apache":
        raise NotImplementedError("Apache Hadoop/Spark integration needs pyarrow or SparkSession setup.")

    else:
        raise ValueError("Unsupported database type")
