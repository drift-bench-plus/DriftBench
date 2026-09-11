"""MySQL executor: the authoritative engine for DBBench ground truth.

Why a container and not sqlite: DBBench creates every column as TEXT and inserts
``str(value)`` for everything.  MySQL coerces TEXT numerically in comparisons
(``'100' > 20`` is a numeric test), while SQLite compares by storage class, so
``WHERE textcol > 5`` is true for *every* row.  Counting, ranking and aggregation answers
would silently differ.  Reusing the benchmark's own engine is the only way to keep its
evaluation semantics intact (north-star #2).

Isolation:
  * one long-lived server container for the whole run
  * one DATABASE per environment, dropped on close
  * a mutating recipe always gets a freshly reloaded database, because write-task ground
    truth is the post-execution table state -- running two write candidates against one
    database would make every hash after the first quietly wrong
"""

from __future__ import annotations

import logging
import subprocess
import time
import uuid

from .base import BaseSession

log = logging.getLogger(__name__)


class MySQLSession(BaseSession):
    def __init__(self, executor: MySQLExecutor, env_spec: dict) -> None:
        super().__init__(executor, env_spec)
        # ⚠ unique per session: two sessions on one environment must not collide, because
        # _materialize() begins with DROP DATABASE -- a scratch session sharing the name
        # would destroy the episode's database.
        self.db_name = f"it_{env_spec['env_id'][:24].replace('-', '_')}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _dedupe_columns(columns: list[str]) -> list[str]:
        """Some shipped tables repeat a column name; MySQL rejects that.

        The first occurrence keeps the original name so reference SQL referring to it
        still resolves; later duplicates get a suffix.
        """
        seen: dict[str, int] = {}
        out = []
        for c in columns:
            seen[c] = seen.get(c, 0) + 1
            out.append(c if seen[c] == 1 else f"{c}__{seen[c]}")
        return out

    def _materialize(self) -> None:
        cur = self.executor.cursor()
        cur.execute(f"DROP DATABASE IF EXISTS `{self.db_name}`")
        cur.execute(f"CREATE DATABASE `{self.db_name}`")
        cur.execute(f"USE `{self.db_name}`")
        for table in self.env_spec["tables"]:
            name = table["table_name"]
            columns = self._dedupe_columns(list(table["columns"]))
            width = len(columns)
            cols = ",".join(f"`{c}` TEXT" for c in columns)
            colnames = ",".join(f"`{c}`" for c in columns)
            cur.execute(f"CREATE TABLE IF NOT EXISTS `{name}` ({cols})")
            # rows whose arity disagrees with the header are padded/truncated rather than
            # aborting the whole environment (a handful of shipped tables are ragged)
            rows = [(list(r) + [""] * width)[:width] for r in table["rows"]]
            if rows:
                placeholders = ",".join("(" + ",".join(["%s"] * width) + ")" for _ in rows)
                data = tuple(str(v) for row in rows for v in row)
                cur.execute(f"INSERT INTO `{name}` ({colnames}) VALUES {placeholders}", data)
        self.executor.conn.commit()
        cur.close()

    def _execute(self, recipe):
        """Run one statement; return rows for reads, rowcount for writes."""
        sql = recipe["sql"] if isinstance(recipe, dict) else recipe
        cur = self.executor.cursor()
        try:
            cur.execute(f"USE `{self.db_name}`")
            cur.execute(sql)
            if cur.with_rows:
                return cur.fetchall()
            self.executor.conn.commit()
            return []
        finally:
            cur.close()

    def scalar(self, sql: str):
        rows = self._execute({"sql": sql})
        return rows[0][0] if rows and rows[0] else None

    def close(self) -> None:
        if not self._closed:
            try:
                cur = self.executor.cursor()
                cur.execute(f"DROP DATABASE IF EXISTS `{self.db_name}`")
                cur.close()
            except Exception as exc:
                log.debug("drop database failed: %s", exc)
        super().close()


class MySQLExecutor:
    name = "mysql_docker"

    def __init__(self, config: dict) -> None:
        d = config["docker"]
        self.image = d["mysql_image"]
        self.port = int(d["mysql_port"])
        self.password = d["mysql_password"]
        self.label = d["label"]
        self.timeout = int(d.get("sql_timeout_s", 60))
        self.container = f"it-mysql-{uuid.uuid4().hex[:8]}"
        self.conn = None
        self._started = False

    # -- container lifecycle --------------------------------------------------
    def _docker(self, *args, check=True, capture=True):
        return subprocess.run(["docker", *args], check=check,
                              capture_output=capture, text=True, timeout=300)

    def _start(self) -> None:
        if self._started:
            return
        self._docker(
            "run", "-d", "--rm",
            "--label", f"{self.label}=1",
            "--name", self.container,
            "-e", f"MYSQL_ROOT_PASSWORD={self.password}",
            "-p", f"127.0.0.1:{self.port}:3306",
            self.image,
        )
        self._started = True
        deadline = time.time() + 180
        while time.time() < deadline:
            r = self._docker("exec", self.container, "mysqladmin", "ping",
                             "-uroot", f"-p{self.password}", "--silent", check=False)
            if r.returncode == 0:
                break
            time.sleep(2)
        else:
            raise RuntimeError(f"mysql container {self.container} never became ready")
        self._connect()

    def _connect(self) -> None:
        import mysql.connector

        last = None
        for _ in range(30):
            try:
                self.conn = mysql.connector.connect(
                    host="127.0.0.1", port=self.port, user="root",
                    password=self.password, connection_timeout=self.timeout,
                    autocommit=False,
                )
                return
            except Exception as exc:  # server accepts TCP slightly after mysqladmin ping
                last = exc
                time.sleep(2)
        raise RuntimeError(f"cannot connect to mysql: {last}")

    def cursor(self):
        """A live cursor, reconnecting if the server dropped the connection.

        Long runs open and drop hundreds of databases; the connection occasionally goes
        away and would otherwise fail a whole environment.
        """
        if self.conn is None:
            self._start()
        try:
            if not self.conn.is_connected():
                self.conn.reconnect(attempts=3, delay=1)
        except Exception:
            self._connect()
        return self.conn.cursor()

    def open(self, env_spec: dict) -> MySQLSession:
        self._start()
        return MySQLSession(self, env_spec)

    def shutdown(self) -> None:
        try:
            if self.conn is not None:
                self.conn.close()
        finally:
            if self._started:
                self._docker("rm", "-f", self.container, check=False)
                self._started = False


def drop_stale_databases(executor: MySQLExecutor) -> int:
    """Drop leftover `it_*` databases.

    Unique per-session names mean a crashed run no longer self-cleans by reopening the same
    name, and janitor() only removes containers.
    """
    cur = executor.cursor()
    cur.execute("SHOW DATABASES")
    names = [r[0] for r in cur.fetchall() if str(r[0]).startswith("it_")]
    for n in names:
        cur.execute(f"DROP DATABASE IF EXISTS `{n}`")
    cur.close()
    return len(names)


def janitor(label: str = "intent-graph") -> int:
    """Remove every container this pipeline created. Safe to call anytime."""
    ids = subprocess.run(["docker", "ps", "-aq", "--filter", f"label={label}=1"],
                         capture_output=True, text=True).stdout.split()
    for cid in ids:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
    return len(ids)
