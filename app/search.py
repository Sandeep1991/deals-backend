from __future__ import annotations

from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery

from app.config import Settings, get_settings
from app.models import Ad, SearchResultItem


class SearchNotConfiguredError(RuntimeError):
    pass


class SearchService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: SearchClient | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.azure_search_endpoint and self.settings.azure_search_api_key)

    @property
    def client(self) -> SearchClient:
        if not self.is_configured:
            raise SearchNotConfiguredError(
                "Azure AI Search is not configured. Set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY."
            )
        if self._client is None:
            self._client = SearchClient(
                endpoint=self.settings.azure_search_endpoint,
                index_name=self.settings.azure_search_index,
                credential=AzureKeyCredential(self.settings.azure_search_api_key),
            )
        return self._client

    def _meaning_terms(self) -> set[str]:
        return {t.strip().lower() for t in self.settings.meaning_query_terms.split(",") if t.strip()}

    def _is_meaning_query(self, query: str) -> bool:
        tokens = set(query.lower().replace(",", " ").split())
        return bool(tokens & self._meaning_terms())

    def _to_ad(self, doc: dict[str, Any]) -> Ad:
        return Ad(
            id=str(doc["id"]),
            title=doc.get("title", ""),
            description=doc.get("description", ""),
            category=doc.get("category", ""),
            keywords=doc.get("keywords", ""),
            price=doc.get("price", ""),
            url=doc.get("url", ""),
            merchant=doc.get("merchant") or "",
            network=doc.get("network") or "",
            brand=doc.get("brand") or "",
            source_key=doc.get("source_key") or "",
            expires_at=doc.get("expires_at") or "",
        )

    def _to_result(self, doc: dict[str, Any]) -> SearchResultItem:
        search_score = float(doc.get("@search.score", 0))
        reranker_score = float(doc.get("@search.rerankerScore", 0))
        return SearchResultItem(
            ad=self._to_ad(doc),
            score=reranker_score or search_score,
            keyword_score=search_score,
            semantic_score=reranker_score,
        )

    def _passes_threshold(self, doc: dict[str, Any], min_score: float | None = None) -> bool:
        search_score = float(doc.get("@search.score", 0))
        reranker_score = float(doc.get("@search.rerankerScore", 0))
        min_reranker = min_score if min_score is not None else self.settings.min_reranker_score
        min_search = min_score if min_score is not None else self.settings.min_search_score
        if reranker_score > 0:
            return reranker_score >= min_reranker
        return search_score >= min_search

    def _vector_field(self) -> str | None:
        field = self.settings.azure_search_vector_field.strip()
        return field or None

    def _vector_queries(self, query: str) -> list[VectorizableTextQuery] | None:
        field = self._vector_field()
        if not field:
            return None
        return [
            VectorizableTextQuery(
                text=query,
                k_nearest_neighbors=self.settings.search_top_k,
                fields=field,
            )
        ]

    def _run_search(self, **kwargs: Any) -> list[dict[str, Any]]:
        vector_queries = kwargs.pop("vector_queries", None)
        if vector_queries:
            kwargs["vector_queries"] = vector_queries
        return list(self.client.search(**kwargs))

    def search(
        self,
        query: str,
        limit: int | None = None,
        merchant: str | None = None,
        min_score: float | None = None,
    ) -> list[SearchResultItem]:
        limit = limit or self.settings.search_result_limit
        query = query.strip()
        if not query:
            return []

        use_hybrid = self._is_meaning_query(query)

        kwargs: dict[str, Any] = {
            "search_text": query,
            "select": [
                "id", "title", "description", "category", "keywords", "price", "url",
                "merchant", "network", "brand", "source_key", "expires_at",
            ],
            "top": self.settings.search_top_k,
            "search_fields": ["title", "description", "keywords", "category", "merchant", "brand"],
        }

        if merchant:
            safe = merchant.replace("'", "''")
            kwargs["filter"] = f"merchant eq '{safe}'"

        if use_hybrid:
            kwargs["query_type"] = "semantic"
            kwargs["semantic_configuration_name"] = self.settings.azure_search_semantic_config
            vector_queries = self._vector_queries(query)
        else:
            kwargs["query_type"] = "simple"
            vector_queries = None

        docs = self._run_search(**kwargs, vector_queries=vector_queries)

        threshold = min_score if min_score is not None else None
        filtered = [
            doc for doc in docs
            if self._passes_threshold(doc, min_score=threshold)
        ]
        if not filtered and docs and not use_hybrid:
            return self._hybrid_search(query, limit, merchant=merchant, min_score=min_score)

        results = [self._to_result(doc) for doc in filtered[:limit]]
        return results

    def _hybrid_search(
        self,
        query: str,
        limit: int,
        merchant: str | None = None,
        min_score: float | None = None,
    ) -> list[SearchResultItem]:
        kwargs: dict[str, Any] = {
            "search_text": query,
            "query_type": "semantic",
            "semantic_configuration_name": self.settings.azure_search_semantic_config,
            "select": [
                "id", "title", "description", "category", "keywords", "price", "url",
                "merchant", "network", "brand", "source_key", "expires_at",
            ],
            "search_fields": ["title", "description", "keywords", "category", "merchant", "brand"],
            "top": self.settings.search_top_k,
        }
        if merchant:
            safe = merchant.replace("'", "''")
            kwargs["filter"] = f"merchant eq '{safe}'"
        docs = self._run_search(**kwargs, vector_queries=self._vector_queries(query))
        threshold = min_score if min_score is not None else None
        filtered = [
            doc for doc in docs
            if self._passes_threshold(doc, min_score=threshold)
        ]
        return [self._to_result(doc) for doc in filtered[:limit]]

    def upsert_ads(self, ads: list[Ad]) -> list[str]:
        documents = [ad.model_dump() for ad in ads]
        results = self.client.merge_or_upload_documents(documents)
        failed = [r for r in results if not r.succeeded]
        if failed:
            messages = ", ".join(f"{r.key}: {r.error_message}" for r in failed)
            raise RuntimeError(f"Failed to upsert ads: {messages}")
        return [ad.id for ad in ads]

    def delete_ad(self, ad_id: str) -> None:
        results = self.client.delete_documents(documents=[{"id": ad_id}])
        failed = [r for r in results if not r.succeeded]
        if failed:
            raise RuntimeError(f"Failed to delete ad {ad_id}: {failed[0].error_message}")
