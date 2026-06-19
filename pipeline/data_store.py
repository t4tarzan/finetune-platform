"""
DuckDB Data Store — columnar storage layer for training data.

Replaces flat JSONL with DuckDB-backed columnar storage:
  - Parquet import/export (GB-scale, compressed, columnar)
  - SQL-based filtering, sampling, and aggregation
  - Schema auto-detection from CSV/Parquet/JSONL
  - Training dataset generation via SQL queries
  - Customer DB schema integration

Usage:
  from pipeline.data_store import DataStore
  ds = DataStore("data/training.db")
  ds.import_parquet("customer_data.parquet")
  rows = ds.query("SELECT * FROM data WHERE domain = 'medical' LIMIT 100")
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import duckdb
import yaml


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


SCHEMA_TEMPLATES = {
    "training": {
        "id": "VARCHAR",
        "question": "VARCHAR",
        "reference_answer": "VARCHAR",
        "context": "VARCHAR",
        "domain": "VARCHAR",
        "source": "VARCHAR",
        "created_at": "TIMESTAMP",
        "verified": "BOOLEAN",
        "consensus_score": "FLOAT",
    },
    "customer": {
        "id": "VARCHAR",
        "customer_id": "VARCHAR",
        "schema_name": "VARCHAR",
        "table_name": "VARCHAR",
        "columns": "VARCHAR",  # JSON array of column definitions
        "row_count": "BIGINT",
        "imported_at": "TIMESTAMP",
    },
    "inference_log": {
        "id": "VARCHAR",
        "model_name": "VARCHAR",
        "prompt": "VARCHAR",
        "response": "VARCHAR",
        "latency_ms": "INTEGER",
        "tokens_generated": "INTEGER",
        "created_at": "TIMESTAMP",
    },
}


class DataStore:
    """DuckDB-backed columnar data store for training and inference data."""

    def __init__(self, db_path: str = None, config: Optional[dict] = None):
        self.config = config or load_config()
        data_dir = self.config.get("paths", {}).get("data", "data")
        self.db_path = db_path or os.path.join(data_dir, "training.db")
        self._ensure_schema()

    def _ensure_schema(self):
        """Create tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        con = duckdb.connect(self.db_path)
        for name, columns in SCHEMA_TEMPLATES.items():
            col_defs = ", ".join(f"{col} {typ}" for col, typ in columns.items())
            con.execute(f"CREATE TABLE IF NOT EXISTS {name} ({col_defs})")
        con.close()

    def connect(self):
        """Get a DuckDB connection (caller must close)."""
        return duckdb.connect(self.db_path)

    # ── Import ──────────────────────────────────────────────────

    def import_jsonl(self, path: str, table: str = "training", domain: str = "general") -> int:
        """Import a JSONL file into a DuckDB table. Returns row count."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        con = self.connect()
        try:
            # Load JSONL into a temp table, then insert into main
            con.execute(f"""
                CREATE TEMP TABLE _tmp AS
                SELECT * FROM read_json_auto('{path}')
            """)
            row_count = con.execute("SELECT COUNT(*) FROM _tmp").fetchone()[0]

            now = datetime.now().isoformat()
            con.execute(f"""
                INSERT INTO {table}
                SELECT *,
                    '{domain}' AS domain,
                    '{path}' AS source,
                    '{now}'::TIMESTAMP AS created_at
                FROM _tmp
            """)
            con.execute("DROP TABLE IF EXISTS _tmp")
            print(f"  Imported {row_count} rows from {path} into '{table}'")
            return row_count
        finally:
            con.close()

    def import_parquet(self, path: str, table: str = "training", domain: str = "general") -> int:
        """Import a Parquet file. Returns row count."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        con = self.connect()
        try:
            con.execute(f"""
                CREATE TEMP TABLE _tmp AS
                SELECT * FROM read_parquet('{path}')
            """)
            row_count = con.execute("SELECT COUNT(*) FROM _tmp").fetchone()[0]

            now = datetime.now().isoformat()
            con.execute(f"""
                INSERT INTO {table}
                SELECT *,
                    '{domain}' AS domain,
                    '{path}' AS source,
                    '{now}'::TIMESTAMP AS created_at
                FROM _tmp
            """)
            con.execute("DROP TABLE IF EXISTS _tmp")
            print(f"  Imported {row_count} rows from {path} into '{table}'")
            return row_count
        finally:
            con.close()

    def import_csv(self, path: str, table: str = "training", domain: str = "general") -> int:
        """Import a CSV file. Auto-detects schema. Returns row count."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        con = self.connect()
        try:
            con.execute(f"""
                CREATE TEMP TABLE _tmp AS
                SELECT * FROM read_csv_auto('{path}')
            """)
            row_count = con.execute("SELECT COUNT(*) FROM _tmp").fetchone()[0]

            now = datetime.now().isoformat()
            con.execute(f"""
                INSERT INTO {table}
                SELECT *,
                    '{domain}' AS domain,
                    '{path}' AS source,
                    '{now}'::TIMESTAMP AS created_at
                FROM _tmp
            """)
            con.execute("DROP TABLE IF EXISTS _tmp")
            print(f"  Imported {row_count} rows from {path} into '{table}'")
            return row_count
        finally:
            con.close()

    # ── Export ──────────────────────────────────────────────────

    def export_parquet(self, query: str, output_path: str):
        """Export query results to Parquet."""
        con = self.connect()
        try:
            con.execute(f"COPY ({query}) TO '{output_path}' (FORMAT PARQUET)")
            row_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path}')").fetchone()[0]
            size = os.path.getsize(output_path)
            print(f"  Exported {row_count} rows → {output_path} ({size/1e6:.1f} MB)")
            return row_count
        finally:
            con.close()

    def export_jsonl(self, query: str, output_path: str):
        """Export query results to JSONL."""
        con = self.connect()
        try:
            con.execute(f"COPY ({query}) TO '{output_path}' (FORMAT JSON)"
                        if not output_path.endswith('.jsonl') else
                        f"COPY ({query}) TO '{output_path}' (FORMAT JSON)")
            row_count = sum(1 for _ in open(output_path) if _.strip())
            print(f"  Exported {row_count} rows → {output_path}")
            return row_count
        finally:
            con.close()

    # ── Query ───────────────────────────────────────────────────

    def query(self, sql: str) -> list[dict]:
        """Execute a SQL query and return results as dicts."""
        con = self.connect()
        try:
            result = con.execute(sql)
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        finally:
            con.close()

    def count(self, table: str = "training", where: str = "1=1") -> int:
        """Count rows in a table with optional WHERE clause."""
        con = self.connect()
        try:
            return con.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()[0]
        finally:
            con.close()

    def list_tables(self) -> list[str]:
        """List all tables in the database."""
        con = self.connect()
        try:
            rows = con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            con.close()

    def describe(self, table: str) -> list[dict]:
        """Describe a table's schema."""
        con = self.connect()
        try:
            rows = con.execute(f"DESCRIBE {table}").fetchall()
            return [{"column": r[0], "type": r[1], "null": r[2], "default": r[3]} for r in rows]
        finally:
            con.close()

    # ── Training data generation ────────────────────────────────

    def generate_training_set(
        self,
        sql: str,
        output_path: str = "data/verified_train.jsonl",
        question_col: str = "question",
        answer_col: str = "reference_answer",
        context_col: Optional[str] = "context",
        format: str = "jsonl",
    ) -> int:
        """
        Generate a training-ready JSONL/Parquet file from a SQL query.
        Maps SQL columns to training format.
        """
        if context_col:
            mapping = f"SELECT {question_col} AS prompt, {answer_col} AS completion, {context_col}"
        else:
            mapping = f"SELECT {question_col} AS prompt, {answer_col} AS completion"

        full_sql = f"""
            SELECT {question_col} AS prompt, {answer_col} AS completion
            {", " + context_col if context_col else ""}
            FROM ({sql}) AS _src
            WHERE {question_col} IS NOT NULL AND {answer_col} IS NOT NULL
        """

        if format == "parquet":
            return self.export_parquet(full_sql, output_path)
        else:
            return self.export_jsonl(full_sql, output_path)

    # ── Customer schema integration ─────────────────────────────

    def register_customer_schema(
        self,
        customer_id: str,
        schema_name: str,
        tables: dict[str, list[dict]],
    ):
        """
        Register a customer's database schema for automated data ingestion.

        Args:
            customer_id: Unique customer identifier
            schema_name: e.g. "medical_billing_db"
            tables: {table_name: [{"column": "col", "type": "VARCHAR", "description": "..."}]}
        """
        con = self.connect()
        try:
            now = datetime.now().isoformat()
            for table_name, columns in tables.items():
                col_json = json.dumps(columns)
                con.execute(f"""
                    INSERT INTO customer (id, customer_id, schema_name, table_name, columns, row_count, imported_at)
                    VALUES ('{customer_id}_{table_name}', '{customer_id}', '{schema_name}',
                            '{table_name}', '{col_json}', 0, '{now}'::TIMESTAMP)
                """)
            print(f"  Registered schema '{schema_name}' for customer '{customer_id}' "
                  f"({len(tables)} tables)")
        finally:
            con.close()

    def generate_customer_training_sql(
        self,
        customer_id: str,
        question_table: str,
        question_column: str,
        answer_table: str,
        answer_column: str,
        join_column: str,
    ) -> str:
        """
        Auto-generate a SQL query for customer training data based on their schema.
        """
        sql = f"""
            SELECT q.{question_column} AS prompt,
                   a.{answer_column} AS completion
            FROM {question_table} q
            JOIN {answer_table} a ON q.{join_column} = a.{join_column}
            WHERE q.{question_column} IS NOT NULL
              AND a.{answer_column} IS NOT NULL
              AND LENGTH(q.{question_column}) > 10
              AND LENGTH(a.{answer_column}) > 5
            ORDER BY random()
        """
        return sql

    # ── Stats ───────────────────────────────────────────────────

    def stats(self) -> dict:
        """Get database statistics."""
        con = self.connect()
        try:
            tables = self.list_tables()
            result = {}
            for t in tables:
                count = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                try:
                    size = con.execute(
                        f"SELECT COUNT(DISTINCT domain) FROM {t}"
                    ).fetchone()[0]
                except Exception:
                    size = 0
                result[t] = {"rows": count, "domains": size}
            return result
        finally:
            con.close()

    def file_size(self) -> str:
        """Get database file size as human-readable string."""
        path = self.db_path
        if os.path.exists(path):
            size = os.path.getsize(path)
            for unit in ["B", "KB", "MB", "GB"]:
                if size < 1024:
                    return f"{size:.1f} {unit}"
                size /= 1024
            return f"{size:.1f} TB"
        return "0 B"


if __name__ == "__main__":
    # Quick test
    ds = DataStore("data/test_duck.db")
    print(f"Tables: {ds.list_tables()}")
    print(f"File size: {ds.file_size()}")

    # Test JSONL import
    if os.path.exists("data/example_train.jsonl"):
        ds.import_jsonl("data/example_train.jsonl", domain="test")
        print(f"Training rows: {ds.count('training')}")
        print(f"Schema: {ds.describe('training')}")

    # Test customer schema registration
    ds.register_customer_schema("cust-001", "medical_billing", {
        "claims": [
            {"column": "claim_id", "type": "VARCHAR", "description": "Claim identifier"},
            {"column": "diagnosis_code", "type": "VARCHAR", "description": "ICD-10 code"},
        ],
        "providers": [
            {"column": "npi", "type": "VARCHAR", "description": "Provider NPI number"},
            {"column": "specialty", "type": "VARCHAR", "description": "Medical specialty"},
        ],
    })
    print(f"Stats: {ds.stats()}")
