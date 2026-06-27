"""
File Component — IBM Cloud Object Storage integration
────────────────────────────────────────────────────────────────
Handles:
  • User portfolio file uploads and downloads
  • Financial report storage
  • Investment dataset management
  • Local file fallback for development
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO

import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)


class FileComponent:
    """
    Abstracted file storage that works with either
    IBM Cloud Object Storage (COS) or local disk.
    """

    def __init__(self) -> None:
        cfg = get_settings().ibm_cloud
        self._cfg = cfg
        self._cos_client = None
        self._local_root = Path("./data/uploads")
        self._local_root.mkdir(parents=True, exist_ok=True)
        self._backend = self._init_cos()

    def _init_cos(self) -> str:
        if not self._cfg.cos_api_key:
            logger.info("file_component.backend", type="local_disk")
            return "local"
        try:
            import ibm_boto3
            from ibm_botocore.client import Config

            self._cos_client = ibm_boto3.client(
                "s3",
                ibm_api_key_id=self._cfg.cos_api_key,
                ibm_service_instance_id=self._cfg.cos_instance_crn,
                config=Config(signature_version="oauth"),
                endpoint_url=self._cfg.cos_endpoint,
            )
            logger.info("file_component.backend", type="ibm_cos", bucket=self._cfg.cos_bucket_name)
            return "cos"
        except Exception as exc:  # noqa: BLE001
            logger.warning("cos.init_failed", error=str(exc), fallback="local_disk")
            return "local"

    # ─────────────────────────────────────────
    #  Core CRUD
    # ─────────────────────────────────────────

    def upload(self, data: bytes | BinaryIO, key: str, content_type: str = "application/octet-stream") -> str:
        """Upload a file. Returns storage key / path."""
        if isinstance(data, bytes):
            data = io.BytesIO(data)

        if self._backend == "cos":
            self._cos_client.upload_fileobj(
                data,
                self._cfg.cos_bucket_name,
                key,
                ExtraArgs={"ContentType": content_type},
            )
            logger.info("file.uploaded", backend="cos", key=key)
        else:
            dest = self._local_root / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data.read() if hasattr(data, "read") else data)
            logger.info("file.uploaded", backend="local", path=str(dest))
        return key

    def download(self, key: str) -> bytes:
        """Download a file by key. Returns raw bytes."""
        if self._backend == "cos":
            buf = io.BytesIO()
            self._cos_client.download_fileobj(self._cfg.cos_bucket_name, key, buf)
            return buf.getvalue()
        else:
            return (self._local_root / key).read_bytes()

    def delete(self, key: str) -> bool:
        """Delete a file. Returns True if successful."""
        try:
            if self._backend == "cos":
                self._cos_client.delete_object(Bucket=self._cfg.cos_bucket_name, Key=key)
            else:
                (self._local_root / key).unlink(missing_ok=True)
            logger.info("file.deleted", key=key)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("file.delete_failed", key=key, error=str(exc))
            return False

    def list_files(self, prefix: str = "") -> list[dict[str, Any]]:
        """List stored files, optionally filtered by prefix."""
        if self._backend == "cos":
            response = self._cos_client.list_objects_v2(
                Bucket=self._cfg.cos_bucket_name,
                Prefix=prefix,
            )
            return [
                {
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat(),
                }
                for obj in response.get("Contents", [])
            ]
        else:
            root = self._local_root / prefix if prefix else self._local_root
            files = []
            for fp in root.rglob("*"):
                if fp.is_file():
                    stat = fp.stat()
                    files.append({
                        "key": str(fp.relative_to(self._local_root)),
                        "size": stat.st_size,
                        "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    })
            return files

    def exists(self, key: str) -> bool:
        if self._backend == "cos":
            try:
                self._cos_client.head_object(Bucket=self._cfg.cos_bucket_name, Key=key)
                return True
            except Exception:  # noqa: BLE001
                return False
        return (self._local_root / key).exists()

    # ─────────────────────────────────────────
    #  Convenience methods for finance domain
    # ─────────────────────────────────────────

    def save_portfolio(self, user_id: str, portfolio_data: dict) -> str:
        """Persist a user's portfolio as JSON."""
        key = f"portfolios/{user_id}/portfolio_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        data = json.dumps(portfolio_data, indent=2, default=str).encode()
        return self.upload(data, key, content_type="application/json")

    def load_portfolio(self, user_id: str) -> dict | None:
        """Load the most recent portfolio for a user."""
        files = sorted(
            [f for f in self.list_files(f"portfolios/{user_id}/") if f["key"].endswith(".json")],
            key=lambda x: x["last_modified"],
            reverse=True,
        )
        if not files:
            return None
        raw = self.download(files[0]["key"])
        return json.loads(raw)

    def save_report(self, report_type: str, report_data: dict, user_id: str) -> str:
        """Save a generated investment report."""
        key = f"reports/{user_id}/{report_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        data = json.dumps(report_data, indent=2, default=str).encode()
        return self.upload(data, key, content_type="application/json")

    def upload_user_document(self, user_id: str, filename: str, content: bytes) -> str:
        """Upload a user-provided financial document for RAG ingestion."""
        ext = Path(filename).suffix.lower()
        key = f"documents/{user_id}/{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{filename}"
        return self.upload(content, key)
