"""
reranker.py — cross-encoder reranker через sentence-transformers.

Архитектурное отличие от первого этапа поиска (vector search):

  bi-encoder (что у нас в pgvector):
      embed(query) → vec_q
      embed(chunk) → vec_d           ← вектора независимы
      score = cosine(vec_q, vec_d)
      Быстро, можно заранее посчитать vec_d и индексировать.

  cross-encoder (этот модуль):
      model.predict([query, chunk]) → score
      Модель видит query И chunk ВМЕСТЕ через cross-attention внутри
      трансформера. Токены q и d сравниваются между собой напрямую.
      Точнее, но нельзя заранее «закодировать» документ — каждая пара
      проходит модель целиком.

Поэтому reranker — это **второй этап**: на retrieve берём top-15-20 по
быстрому bi-encoder/BM25, потом cross-encoder переранжирует только их.

Модель: `BAAI/bge-reranker-v2-m3` — мультиязычная (русский поддерживается
нативно), та же «семья» что у нашего embedding bge-m3. Архитектура XLM-RoBERTa
+ pooling. На выходе один логит, мы применяем sigmoid → score в [0, 1].

При первой инициализации модель ~570 МБ скачивается с HuggingFace Hub в
~/.cache/huggingface/hub. На следующих запусках загружается из кэша мгновенно.

Работа модели на macOS:
  - На Apple Silicon (M1/M2/M3) sentence-transformers сам выбирает MPS
    backend через PyTorch — это GPU-ускорение Apple. На паре десятков
    чанков почти мгновенно.
  - На Intel / без GPU — CPU. Тоже терпимо: один батч из 15 пар занимает
    100-300 мс на современном CPU.
"""

from __future__ import annotations

import threading
from dataclasses import replace

from sentence_transformers import CrossEncoder

from rag.vector_store import RetrievedChunk


# Имя модели на HuggingFace Hub. Если хочется заменить — можно положить
# в .env, но для учебного проекта хардкод проще.
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker:
    """
    Pointwise reranker на cross-encoder модели.

    Метод rerank принимает кандидатов от первого этапа поиска
    (vector/text/hybrid) и переранжирует их через предсказание модели
    на каждой паре (query, chunk).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        # CrossEncoder лениво загружает модель при первом вызове, но
        # фактически конструктор уже скачивает/загружает в RAM. Это удобно:
        # после __init__ модель готова, первый запрос не «прогревает».
        #
        # max_length=512 — стандартное окно для bge-reranker. Длинные пары
        # модель сама обрежет (truncation). Наши чанки до 500 символов —
        # помещаются с запасом.
        #
        # activation_fn=None оставляем дефолтным — модель сама применит
        # sigmoid на финальном логите, получим score в [0, 1].
        self._model = CrossEncoder(model_name, max_length=512)
        self._model_name = model_name
        # На случай если когда-нибудь будем вызывать из нескольких потоков
        # одновременно — PyTorch внутри может быть не thread-safe.
        # Сейчас один сервер, один запрос за раз — лок просто страховка.
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Прогоняет все пары (query, chunk.content) через модель одним батчем,
        записывает результат в reranker_score, сохраняет original_rank
        (позицию ДО переранжирования), сортирует по убыванию score.

        Один батч на все пары — это важная оптимизация: модель тогда
        обрабатывает их параллельно через GPU/SIMD, а не один за другим.
        """
        if not chunks:
            return []

        pairs = [(query, c.content) for c in chunks]
        with self._lock:
            # predict возвращает numpy.ndarray[float] длиной len(pairs).
            # show_progress_bar=False — иначе sentence-transformers рисует
            # tqdm-progressbar в stdout, нам это в логе сервера не нужно.
            scores = self._model.predict(pairs, show_progress_bar=False)

        enriched = [
            replace(c, reranker_score=float(s), original_rank=i)
            for i, (c, s) in enumerate(zip(chunks, scores), start=1)
        ]
        enriched.sort(
            key=lambda c: (c.reranker_score or 0.0),
            reverse=True,
        )
        if top_k is not None:
            enriched = enriched[:top_k]
        return enriched

    # Lifecycle-методы — у sentence_transformers нет «явного close»,
    # модель будет очищена сборщиком мусора при выгрузке процесса.
    def close(self) -> None:
        pass

    def __enter__(self) -> "CrossEncoderReranker":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
