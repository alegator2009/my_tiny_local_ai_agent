from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx

from ..config import effective_mcp_servers, effective_primary_context_window_size, load_app_config
from ..db import execute, fetch_all, fetch_one, utcnow_iso
from ..storage import append_transcript_event, write_session_json
from .artifacts import write_file_artifact
from .indexing import estimate_token_count, index_message
from .memory import create_checkpoint, maybe_run_scheduled_wiki_lint, maybe_update_durable_facts, record_turn_entities, update_working_set
from .memory import (
    _recent_messages,
    apply_skill_transition,
    build_skill_prompt_bundle,
    list_skill_states,
    load_skill_state,
    reset_skill_state,
    start_or_resume_skill,
)
from .mcp import MCPError, MCPToolRegistry
from .prompt import assemble_prompt
from .auto_search import (
    AutoSearchResult,
    build_grounded_block as _build_grounded_block_helper,
    record_run as _record_auto_search_run,
    run_auto_search as _run_auto_search,
    should_search as _should_auto_search,
)
from .provider_http import (
    build_payload,
    ensure_callable,
    provider_timeout_seconds,
    resolve_provider_model,
)
from .retrieval import run_retrieval, should_trigger_retrieval
from .sessions import create_next_window, get_last_window, get_session
from .terminal import run_terminal_command
from .workspace import log_workspace_event


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _save_message(
    *,
    session_id: str,
    window_id: str,
    role: str,
    content_text: str,
    message_type: str,
    turn_id: str,
    source: str = "chat",
    content_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    message_id = str(uuid.uuid4())
    timestamp = utcnow_iso()
    token_count = estimate_token_count(content_text)

    execute(
        """
        INSERT INTO messages (
          id, session_id, window_id, turn_id, role, timestamp,
          content_text, content_json, token_count, message_type, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            message_id,
            session_id,
            window_id,
            turn_id,
            role,
            timestamp,
            content_text,
            json.dumps(content_json or {}, ensure_ascii=False),
            token_count,
            message_type,
            source,
        ),
    )

    execute(
        """
        UPDATE sessions
        SET total_message_count = total_message_count + 1,
            total_token_count = total_token_count + ?,
            updated_at = ?
        WHERE id = ?
        """,
        (token_count, timestamp, session_id),
    )

    append_transcript_event(
        session_id,
        {
            "timestamp": timestamp,
            "type": "message",
            "message_id": message_id,
            "window_id": window_id,
            "role": role,
            "message_type": message_type,
            "content_text": content_text,
            "token_count": token_count,
        },
    )

    index_message(
        message_id=message_id,
        session_id=session_id,
        window_id=window_id,
        role=role,
        text=content_text,
        timestamp=timestamp,
    )

    return {
        "id": message_id,
        "session_id": session_id,
        "window_id": window_id,
        "turn_id": turn_id,
        "role": role,
        "timestamp": timestamp,
        "content_text": content_text,
        "content_json": content_json or {},
        "token_count": token_count,
        "message_type": message_type,
        "is_pinned": False,
        "is_anchor": False,
        "artifacts": ((content_json or {}).get("artifacts") if isinstance(content_json, dict) else []) or [],
    }


def _window_usage(session_id: str, window_id: str) -> tuple[int, int, float]:
    window_row = fetch_one("SELECT token_limit FROM windows WHERE id=?", (window_id,))
    saved_token_limit = int(window_row["token_limit"]) if window_row else 128000
    cfg = load_app_config()
    # ``saved_token_limit`` may be stale (created when the session had a
    # different model, or before the provider/model picker existed at
    # all — the legacy single-config shape defaulted to 1).  Resolving
    # the current model's window gives us the authoritative upper
    # bound; we then take ``max(saved, current)`` so windows only grow
    # (never silently shrink) when the user picks a smaller model.
    current_model_limit = effective_primary_context_window_size(cfg)
    token_limit = max(saved_token_limit, current_model_limit)
    # ...and finally cap at the configured override if it is lower
    # than the model window.
    override = cfg.model_context_window_size_override
    if override is not None:
        token_limit = min(token_limit, max(1, int(override)))

    used_row = fetch_one(
        "SELECT COALESCE(SUM(token_count), 0) AS used_tokens FROM messages WHERE session_id=? AND window_id=?",
        (session_id, window_id),
    )
    used_tokens = int(used_row["used_tokens"]) if used_row else 0
    used_percent = (used_tokens / max(token_limit, 1)) if token_limit else 0.0
    return token_limit, used_tokens, used_percent


def _provider_timeout_seconds(provider: Any) -> float:
    raw_value = getattr(provider, "request_timeout_sec", 240)
    try:
        timeout_sec = int(raw_value)
    except Exception:
        timeout_sec = 240
    timeout_sec = max(5, min(timeout_sec, 600))
    return float(timeout_sec)


async def _stream_from_provider(
    prompt_messages: list[dict[str, str]],
    *,
    provider_id: str | None = None,
    model_id: str | None = None,
) -> AsyncIterator[str]:
    provider, model = resolve_provider_model(provider_id=provider_id, model_id=model_id)
    if provider is None or model is None or not provider.base_url:
        synthetic = (
            "This is a local fallback response. Configure a provider + model in Settings "
            "to stream from an OpenAI-compatible provider."
        )
        for token in synthetic.split():
            await asyncio.sleep(0.01)
            yield token + " "
        return

    url, headers, payload = build_payload(
        provider=provider,
        model=model,
        messages=prompt_messages,
        stream=True,
    )
    timeout_sec = provider_timeout_seconds(provider)

    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if not data or data == "[DONE]":
                        continue
                    obj = json.loads(data)
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {}).get("content", "")
                    if delta:
                        yield delta
    except Exception as exc:
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc


def _set_pre_rollover_started(window_id: str) -> None:
    execute(
        "UPDATE windows SET pre_rollover_started_at=? WHERE id=? AND pre_rollover_started_at IS NULL",
        (utcnow_iso(), window_id),
    )


def _set_hard_rollover_started(window_id: str) -> None:
    execute(
        "UPDATE windows SET hard_rollover_started_at=? WHERE id=? AND hard_rollover_started_at IS NULL",
        (utcnow_iso(), window_id),
    )


def _needs_retrieval(user_message: str, last_query: str | None) -> bool:
    if should_trigger_retrieval(user_message):
        return True
    if last_query and last_query.strip().lower() == user_message.strip().lower():
        return False
    return False


def _needs_console_tool(user_message: str) -> bool:
    lowered = user_message.lower()
    hints = [
        # English
        "terminal", "console", "run command", "ip address", "network",
        "ls ", "pwd", "cat ", "shell",
        # Spanish
        "terminal ", "consola", "ejecutar", "ip dirección", "red ",
        # French
        "terminal", "console", "exécuter", "adresse ip", "réseau",
        # Portuguese
        "terminal", "console", "executar", "endereço ip", "rede",
        # German
        "terminal", "konsole", "befehl ausführen", "ip-adresse", "netzwerk",
        # Italian
        "terminale", "console", "esegui", "indirizzo ip", "rete",
        # Polish
        "terminal", "konsola", "uruchom", "adres ip", "sieć",
        # Dutch
        "terminal", "console", "voer uit", "ip-adres", "netwerk",
        # Turkish
        "terminal", "konsol", "çalıştır", "ip adresi", "ağ",
        # Vietnamese
        "thiết bị đầu cuối", "bảng điều khiển", "chạy lệnh",
        # Japanese
        "ターミナル", "コンソール", "実行", "ipアドレス", "ネットワーク",
        # Korean
        "터미널", "콘솔", "실행", "ip주소", "네트워크",
        # Chinese
        "终端", "控制台", "运行", "ip地址", "网络",
        # Hindi
        "टर्मिनल", "कंसोल", "चलाओ", "आईपी पता", "नेटवर्क",
        # Arabic
        "الطرفية", "وحدة التحكم", "تشغيل", "عنوان ip", "شبكة",
        # Ukrainian
        "консол", "термінал", "запусти", "виконай", "ip адрес", "мереж",
    ]
    return any(h in lowered for h in hints)


def _needs_file_tool(user_message: str) -> bool:
    lowered = user_message.lower()
    hints = [
        # English
        "save file", "write file", "download file", "artifact",
        "create a file", "save to a file",
        # Spanish
        "guardar archivo", "escribir archivo", "descargar archivo",
        "crear un archivo",
        # French
        "enregistrer le fichier", "écrire le fichier", "télécharger",
        "créer un fichier",
        # Portuguese
        "salvar arquivo", "gravar arquivo", "baixar arquivo",
        "criar um arquivo",
        # German
        "datei speichern", "datei schreiben", "datei herunterladen",
        "datei erstellen",
        # Italian
        "salva file", "scrivi file", "scarica file", "crea un file",
        # Polish
        "zapisz plik", "napisz plik", "pobierz plik", "utwórz plik",
        # Dutch
        "bestand opslaan", "bestand schrijven", "bestand downloaden",
        "bestand maken",
        # Turkish
        "dosya kaydet", "dosya yaz", "dosya indir", "dosya oluştur",
        # Vietnamese
        "lưu tệp", "ghi tệp", "tải tệp", "tạo tệp",
        # Japanese
        "ファイルを保存", "ファイルを書き込む", "ファイルをダウンロード",
        "ファイルを作成",
        # Korean
        "파일 저장", "파일 쓰기", "파일 다운로드", "파일 만들기",
        # Chinese
        "保存文件", "写文件", "下载文件", "创建文件",
        # Hindi
        "फाइल सेव", "फाइल लिखो", "फाइल डाउनलोड", "फाइल बनाओ",
        # Arabic
        "حفظ الملف", "كتابة الملف", "تنزيل الملف", "إنشاء ملف",
        # Ukrainian
        "створи файл", "збережи", "файл", "скачати", "артефакт",
    ]
    if any(h in lowered for h in hints):
        return True
    return bool(re.search(r"\.[a-z0-9]{1,8}\b", lowered))


def _terminal_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "run_terminal_command",
            "description": "Run a terminal command on the local workspace and return stdout/stderr/exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_sec": {"type": "integer", "minimum": 1, "maximum": 120},
                },
                "required": ["command"],
            },
        },
    }


def _file_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write a file to the session workspace and create a downloadable artifact snapshot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path (for example: pages/frogs_page.html).",
                    },
                    "content": {"type": "string", "description": "Text file content."},
                    "content_base64": {"type": "string", "description": "Binary file content encoded as base64."},
                    "encoding": {"type": "string", "description": "Text encoding for content. Default utf-8."},
                    "overwrite": {"type": "boolean", "description": "Overwrite existing file. Default true."},
                    "mime_type": {"type": "string", "description": "Explicit media type for download."},
                    "download_name": {"type": "string", "description": "File name shown to user when downloading."},
                },
                "required": ["path"],
            },
        },
    }


def _terminal_prompt_line() -> str:
    return "- run_terminal_command: execute shell command in workspace and return stdout/stderr/exit code."


def _file_tool_prompt_line() -> str:
    return (
        "- write_file: save requested content into a file and generate downloadable artifact metadata."
    )


def _needs_fresh_or_local_info(user_message: str) -> bool:
    lowered = user_message.lower()

    freshness_hints = [
        # English
        "today", "latest", "current", "fresh", "recent", "now",
        "this week", "this month", "this year",
        # Spanish
        "hoy", "último", "última", "actual", "reciente", "ahora",
        # French
        "aujourd'hui", "dernier", "dernière", "actuel", "récent", "maintenant",
        # Portuguese
        "hoje", "último", "última", "atual", "recente", "agora",
        # German
        "heute", "neueste", "aktuell", "kürzlich", "jetzt",
        # Italian
        "oggi", "ultimo", "ultima", "attuale", "recente", "ora",
        # Polish
        "dzisiaj", "najnowszy", "najnowsza", "aktualny", "ostatni", "teraz",
        # Dutch
        "vandaag", "laatste", "recent", "nu",
        # Turkish
        "bugün", "son", "en son", "güncel", "şimdi",
        # Vietnamese
        "hôm nay", "mới nhất", "hiện tại", "gần đây", "bây giờ",
        # Japanese
        "今日", "最新", "現在", "最近", "今",
        # Korean
        "오늘", "최신", "현재", "최근", "지금",
        # Chinese
        "今天", "最新", "当前", "最近", "现在",
        # Hindi
        "आज", "नवीनतम", "वर्तमान", "हालिया", "अभी",
        # Arabic
        "اليوم", "أحدث", "حالي", "حديث", "الآن",
        # Ukrainian
        "актуальн", "свіж", "зараз", "сьогодні",
    ]
    local_or_review_hints = [
        # English
        "review", "reviews", "rating", "best", "top",
        "recommend", "recommended", "where to buy", "near me",
        "google maps", "instagram", "facebook", "yelp",
        # Spanish
        "reseña", "reseñas", "opinión", "opiniones", "calificación",
        "mejor", "recomendar", "recomendado", "dónde comprar", "cerca de mí",
        # French
        "avis", "évaluation", "note", "meilleur", "top",
        "recommand", "recommandé", "où acheter", "près de moi",
        # Portuguese
        "avaliação", "avaliações", "opinião", "melhor",
        "recomendar", "recomendado", "onde comprar", "perto de mim",
        # German
        "bewertung", "rezension", "beste", "empfehlung", "empfohlen",
        "wo kaufen", "in meiner nähe",
        # Italian
        "recensione", "recensioni", "valutazione", "migliore",
        "consiglia", "consigliato", "dove comprare", "vicino a me",
        # Polish
        "recenzja", "recenzje", "ocena", "najlepszy", "najlepsza",
        "polecam", "polecenie", "gdzie kupić", "blisko mnie",
        # Dutch
        "beoordeling", "beoordelingen", "review", "beste",
        "aanbevelen", "aanbevolen", "waar kopen", "bij mij in de buurt",
        # Turkish
        "yorum", "yorumlar", "değerlendirme", "en iyi",
        "tavsiye", "nereden alınır", "yakınımda",
        # Vietnamese
        "đánh giá", "nhận xét", "tốt nhất", "khuyến nghị",
        "mua ở đâu", "gần tôi",
        # Japanese
        "レビュー", "評価", "口コミ", "最高", "おすすめ",
        "どこで買う", "近く",
        # Korean
        "리뷰", "평가", "추천", "최고", "어디서 사", "근처",
        # Chinese
        "评论", "评分", "推荐", "最好", "在哪买", "附近",
        # Hindi
        "समीक्षा", "रेटिंग", "सर्वश्रेष्ठ", "सुझाव",
        "कहाँ खरीदें", "मेरे पास",
        # Arabic
        "مراجعة", "تقييم", "أفضل", "موصى به", "أين أشتري", "بالقرب مني",
        # Ukrainian
        "відгук", "найкращ", "рекоменд", "де купити", "поруч",
    ]
    place_hints = [
        # English: "in <place>", "near <place>"
        " in ", " near ",
        # Spanish: "en <lugar>", "cerca de"
        " en ", " cerca de ",
        # French: "à <lieu>", "près de"
        " à ", " près de ",
        # Portuguese: "em <lugar>", "perto de"
        " em ", " perto de ",
        # German: "in <ort>", "nahe"
        " in ", " nahe ",
        # Italian: "a <luogo>", "vicino a"
        " a ", " vicino a ",
        # Polish: "w <miejscu>", "blisko"
        " w ", " blisko ",
        # Dutch: "in <plaats>", "dichtbij"
        " in ", " dichtbij ",
        # Turkish: "<yer>'de", "<yer>'a yakın"
        # Vietnamese: "tại <nơi>", "gần"
        " tại ", " gần ",
        # Japanese: "<場所>で", "<場所>の近く"
        # Korean: "<장소>에", "<장소> 근처"
        # Chinese: "在<地方>", "<地方>附近"
        " 在 ", " 附近 ",
        # Hindi: "<जगह> में", "<जगह> के पास"
        # Arabic: "في <مكان>", "بالقرب من"
        # Ukrainian: "у <місто>", "в <місто>", "біля <місто>" + city names
        " у ", " в ", " біля ",
        "дніпр", "київ", "львів", "одес", "харків",
    ]
    local_categories = [
        # English
        "cafe", "restaurant", "cake", "pastry", "bakery",
        "shop", "store", "service",
        # Spanish
        "café", "restaurante", "pastel", "panadería", "tienda",
        # French
        "café", "restaurant", "gâteau", "pâtisserie", "magasin",
        # Portuguese
        "café", "restaurante", "bolo", "padaria", "loja",
        # German
        "café", "restaurant", "kuchen", "bäckerei", "laden", "geschäft",
        # Italian
        "caffè", "ristorante", "torta", "pasticceria", "negozio",
        # Polish
        "kawiarnia", "restauracja", "ciasto", "piekarnia", "sklep",
        # Dutch
        "café", "restaurant", "taart", "bakkerij", "winkel",
        # Turkish
        "kafe", "restoran", "pasta", "fırın", "mağaza",
        # Vietnamese
        "quán cà phê", "nhà hàng", "bánh", "cửa hàng",
        # Japanese
        "カフェ", "レストラン", "ケーキ", "パン屋", "店",
        # Korean
        "카페", "레스토랑", "케이크", "빵집", "가게",
        # Chinese
        "咖啡馆", "餐厅", "蛋糕", "面包店", "商店",
        # Hindi
        "कैफे", "रेस्तरां", "केक", "बेकरी", "दुकान",
        # Arabic
        "مقهى", "مطعم", "كيك", "مخبز", "متجر",
        # Ukrainian
        "кафе", "ресторан", "торт", "кондитер", "магазин", "сервіс",
    ]

    if any(h in lowered for h in freshness_hints):
        return True
    if any(h in lowered for h in local_or_review_hints):
        return True
    if any(h in lowered for h in place_hints) and any(
        h in lowered for h in local_categories
    ):
        return True
    return False


def _wants_tools(user_message: str) -> bool:
    lowered = user_message.lower()
    hints = [
        # English
        "tool", "mcp", "internet", "web", "search",
        "use ", "run ", "check ", "weather", "news",
        "price", "stock", "crypto", "btc", "bitcoin",
        "lookup", "find",
        # Spanish
        "herramienta", "internet", "búsqueda", "buscar",
        "usar ", "ejecutar", "verificar", "clima", "noticias",
        "precio", "cripto", "encontrar",
        # French
        "outil", "internet", "recherche", "chercher",
        "utiliser ", "exécuter", "vérifier", "météo", "actualités",
        "prix", "crypto", "trouver",
        # Portuguese
        "ferramenta", "internet", "pesquisa", "pesquisar",
        "usar ", "executar", "verificar", "clima", "notícias",
        "preço", "cripto", "encontrar",
        # German
        "werkzeug", "internet", "suche", "suchen",
        "benutzen ", "ausführen", "prüfen", "wetter", "nachrichten",
        "preis", "krypto", "finden",
        # Italian
        "strumento", "internet", "ricerca", "cercare",
        "usare ", "eseguire", "verificare", "meteo", "notizie",
        "prezzo", "cripto", "trovare",
        # Polish
        "narzędzie", "internet", "wyszukiwanie", "szukaj",
        "użyj ", "uruchom", "sprawdź", "pogoda", "wiadomości",
        "cena", "krypto", "znajdź",
        # Dutch
        "gereedschap", "internet", "zoeken",
        "gebruik ", "uitvoeren", "controleren", "weer", "nieuws",
        "prijs", "crypto", "vinden",
        # Turkish
        "araç", "internet", "arama", "ara",
        "kullan ", "çalıştır", "kontrol", "hava", "haberler",
        "fiyat", "kripto", "bul",
        # Vietnamese
        "công cụ", "internet", "tìm kiếm", "tìm",
        "sử dụng ", "chạy", "kiểm tra", "thời tiết", "tin tức",
        "giá", "tiền điện tử",
        # Japanese
        "ツール", "インターネット", "検索", "探して",
        "使って", "実行", "確認", "天気", "ニュース",
        "価格", "暗号通貨",
        # Korean
        "도구", "인터넷", "검색", "찾기",
        "사용 ", "실행", "확인", "날씨", "뉴스",
        "가격", "암호화폐",
        # Chinese
        "工具", "互联网", "搜索", "查找",
        "使用 ", "运行", "检查", "天气", "新闻",
        "价格", "加密货币",
        # Hindi
        "उपकरण", "इंटरनेट", "खोज", "खोजो",
        "उपयोग ", "चलाओ", "जांचो", "मौसम", "समाचार",
        "मूल्य", "क्रिप्टो",
        # Arabic
        "أداة", "إنترنت", "بحث", "ابحث",
        "استخدم ", "تشغيل", "تحقق", "طقس", "أخبار",
        "سعر", "تشفير",
        # Ukrainian
        "інструмент", "пошукай", "знайди", "виконай",
        "запусти", "перевір", "погод", "новин", "ціна",
    ]
    return any(h in lowered for h in hints) or _needs_fresh_or_local_info(user_message)


def _with_message_prefix_prompt(user_message: str, message_prefix_prompt: str) -> str:
    prompt = message_prefix_prompt.strip()
    if not prompt:
        return user_message
    return f"{prompt}\n\n{user_message}"


def _safe_json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_inline_tool_args(raw_args: str) -> dict[str, Any]:
    text = raw_args.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    normalized = text
    normalized = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', normalized)
    normalized = normalized.replace("'", '"')
    normalized = re.sub(r",\s*}", "}", normalized)
    try:
        parsed = json.loads(normalized)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_inline_tool_call(content: str) -> tuple[str, dict[str, Any]] | None:
    # Some providers emit pseudo tags like:
    # <tool_call>call:mcp__native_web_search__get_web_search_summaries{query:"..."}
    # instead of proper tool_calls JSON. Support this fallback.
    match = re.search(r"call:\s*([A-Za-z0-9_:-]+)\s*(\{[\s\S]*?\})", content)
    if not match:
        return None
    fn_name = match.group(1).strip()
    args_raw = match.group(2)
    if not fn_name:
        return None
    return fn_name, _parse_inline_tool_args(args_raw)


def _tool_name_from_schema(tool_schema: dict[str, Any]) -> str | None:
    function = tool_schema.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    if not isinstance(name, str):
        return None
    name = name.strip()
    return name or None


def _tool_props_from_schema(tool_schema: dict[str, Any]) -> dict[str, Any]:
    function = tool_schema.get("function")
    if not isinstance(function, dict):
        return {}
    params = function.get("parameters")
    if not isinstance(params, dict):
        return {}
    props = params.get("properties")
    return props if isinstance(props, dict) else {}


def _find_tool_schema_by_name_fragment(tools_schema: list[dict[str, Any]], fragment: str) -> dict[str, Any] | None:
    for schema in tools_schema:
        name = _tool_name_from_schema(schema)
        if name and fragment in name:
            return schema
    return None


def _pick_web_search_tool_schema(tools_schema: list[dict[str, Any]]) -> dict[str, Any] | None:
    named: list[tuple[str, dict[str, Any]]] = []
    for schema in tools_schema:
        name = _tool_name_from_schema(schema)
        if not name:
            continue
        named.append((name, schema))

    for name, schema in named:
        if "__web_search__get_web_search_summaries" in name:
            return schema
    for name, schema in named:
        if "__web_search__full_web_search" in name:
            return schema
    for name, schema in named:
        if "web_search" in name:
            return schema
    return None


def _extract_last_user_query(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def _build_search_fallback_args(tool_schema: dict[str, Any], query: str) -> dict[str, Any]:
    if not query.strip():
        return {}
    props = _tool_props_from_schema(tool_schema)
    if not props:
        return {"query": query}

    out: dict[str, Any] = {}
    for key in ("query", "q", "search_query", "input", "prompt", "question", "text"):
        if key in props:
            out[key] = query
            break
    if not out:
        return {}
    if "limit" in props:
        out["limit"] = 5
    return out


def _truncate_tool_text(text: str, max_chars: int = 3200) -> str:
    value = (text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "\n\n[tool result truncated]"


def _looks_like_empty_search_result(text: str) -> bool:
    lowered = (text or "").lower()
    return "0 results" in lowered or "0 result" in lowered or "no results" in lowered


def _tool_result_to_text(result: Any) -> str:
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if text:
                        text_parts.append(str(text))
            if text_parts:
                return "\n".join(text_parts)
        return json.dumps(result, ensure_ascii=False)
    if isinstance(result, list):
        return json.dumps(result, ensure_ascii=False)
    return str(result)


def _append_artifacts_from_tool_result(result: Any, sink: list[dict[str, Any]]) -> None:
    if not isinstance(result, dict):
        return

    candidates: list[dict[str, Any]] = []
    artifact = result.get("artifact")
    if isinstance(artifact, dict):
        candidates.append(artifact)

    artifacts = result.get("artifacts")
    if isinstance(artifacts, list):
        for item in artifacts:
            if isinstance(item, dict):
                candidates.append(item)

    if not candidates:
        return

    existing_ids = {str(item.get("id")) for item in sink if isinstance(item, dict) and item.get("id")}
    for item in candidates:
        item_id = str(item.get("id") or "")
        if item_id and item_id in existing_ids:
            continue
        if item_id:
            existing_ids.add(item_id)
        sink.append(item)


def _terminal_result_to_text(result: dict[str, Any]) -> str:
    return (
        f"exit_code={result.get('exit_code')}\n"
        f"stdout:\n{result.get('stdout', '')}\n"
        f"stderr:\n{result.get('stderr', '')}"
    )


async def _execute_tool_call(
    *,
    fn_name: str,
    args: dict[str, Any],
    session_id: str,
    session_info: dict[str, Any],
    mcp_registry: MCPToolRegistry | None,
) -> dict[str, Any]:
    workspace_path = session_info.get("workspace_path")
    if fn_name == "write_file":
        result = write_file_artifact(session_id=session_id, session_info=session_info, args=args)
        artifact = result.get("artifact") if isinstance(result, dict) else None
        if isinstance(artifact, dict):
            log_workspace_event(
                session_id=session_id,
                event_type="file_change",
                payload_json={
                    "file": artifact.get("workspace_path"),
                    "workspace_abs_path": artifact.get("workspace_abs_path"),
                    "bytes_written": result.get("bytes_written"),
                    "tool": fn_name,
                },
                summary_text=f"file written: {artifact.get('workspace_path')}",
            )
        return result

    if fn_name == "run_terminal_command":
        command = str(args.get("command", "")).strip()
        timeout_raw = args.get("timeout_sec", 25)
        try:
            timeout_sec = max(1, min(int(timeout_raw), 120))
        except Exception:
            timeout_sec = 25
        if not command:
            return {"ok": False, "error": "empty command"}

        # Hallucinated-command guard. Small local models sometimes
        # invent utilities ("search", "google", "ask", …) that don't
        # exist on the host shell. The web-search MCP is the canonical
        # way to satisfy those intents — we surface the mismatch to
        # the model with a structured error so it can retry through
        # the right tool instead of burning more turns on the same
        # wrong command.
        try:
            from .tool_validation import validate_terminal_command
            cmd_error = validate_terminal_command(command)
            if cmd_error:
                log_workspace_event(
                    session_id=session_id,
                    event_type="terminal_command_rejected",
                    payload_json={"command": command, "reason": cmd_error},
                    summary_text=f"terminal command rejected: {cmd_error}",
                )
                return {
                    "ok": False,
                    "error": cmd_error,
                    "command": command,
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "",
                    "elapsed_ms": 0,
                }
        except Exception:
            # If the validator itself errors out we fall through to
            # the existing execution path rather than block the user.
            pass

        log_workspace_event(
            session_id=session_id,
            event_type="terminal_command",
            payload_json={"command": command, "cwd": workspace_path, "timeout_sec": timeout_sec},
            summary_text=f"terminal command: {command}",
        )
        result = run_terminal_command(command, cwd=workspace_path, timeout_sec=timeout_sec)
        log_workspace_event(
            session_id=session_id,
            event_type="terminal_output",
            payload_json={
                "command": command,
                "exit_code": result.get("exit_code"),
                "ok": result.get("ok"),
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
            },
            summary_text=f"terminal output: exit={result.get('exit_code')}",
        )
        return result

    resolved_mcp_tool_name = mcp_registry.resolve_tool_name(fn_name) if mcp_registry else None
    if mcp_registry and resolved_mcp_tool_name:
        log_workspace_event(
            session_id=session_id,
            event_type="mcp_tool_call",
            payload_json={"tool": resolved_mcp_tool_name, "requested_tool": fn_name, "arguments": args},
            summary_text=f"mcp tool call: {resolved_mcp_tool_name}",
        )
        # SKILL.state bridge: when the model calls a tool that lives on
        # the "skills" MCP server, record the invocation as a tool
        # observation in the persisted skill state. The wrapper.mjs
        # already returns a SKILL.state bundle alongside the tool
        # result, so this is what keeps the persisted state in sync
        # with what the model actually executed.
        skill_state_name: str | None = None
        if mcp_registry:
            try:
                tool = mcp_registry.tools_by_name.get(resolved_mcp_tool_name)
                if tool is not None and getattr(tool, "server_name", None) == "skills":
                    skill_state_name = getattr(tool, "tool_name", None) or fn_name
            except Exception:
                skill_state_name = None
        try:
            result = await mcp_registry.call_tool(resolved_mcp_tool_name, args)
            log_workspace_event(
                session_id=session_id,
                event_type="mcp_tool_result",
                payload_json={"tool": resolved_mcp_tool_name, "requested_tool": fn_name, "result": result},
                summary_text=f"mcp tool result: {resolved_mcp_tool_name}",
            )
            if skill_state_name:
                try:
                    apply_skill_transition(
                        session_id,
                        skill_state_name,
                        transition={"kind": "advance"},
                        observation={
                            "kind": "tool",
                            "source": f"mcp:{fn_name}",
                            "content": _tool_result_to_text(result),
                            "meta": {"requested_tool": fn_name},
                        },
                    )
                except Exception:
                    # Persistence failures are non-fatal: the chat
                    # response must still go through.
                    pass
            return result
        except MCPError as exc:
            err = str(exc)
            log_workspace_event(
                session_id=session_id,
                event_type="mcp_tool_error",
                payload_json={"tool": resolved_mcp_tool_name, "requested_tool": fn_name, "error": err},
                summary_text=f"mcp tool error: {resolved_mcp_tool_name}",
            )
            return {
                "ok": False,
                "error": err,
                "tool": resolved_mcp_tool_name,
                "retryable": "Timeout waiting for MCP response" in err,
            }

    return {"ok": False, "error": f"unknown tool: {fn_name}"}


async def _run_with_tool_loop(
    *,
    prompt_messages: list[dict[str, str]],
    tools_schema: list[dict[str, Any]],
    execute_tool_call: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
    session_id: str,
    window_id: str,
    turn_id: str,
    raw_user_query: str = "",
    expanded_search_query: str = "",
    artifact_sink: list[dict[str, Any]] | None = None,
    tool_event_sink: "asyncio.Queue | None" = None,
    thinking_mode: str = "medium",
    on_tool_call_started: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    provider_id: str | None = None,
    model_id: str | None = None,
) -> str:
    async def _emit_tool_started(fn_name: str, args: dict[str, Any]) -> None:
        # Used by the inline tool-call path.  When we are drained in
        # real-time from stream_chat, on_tool_call_started is None and the
        # surrounding put_nowait already takes care of emission - we just
        # noop here.
        if on_tool_call_started is None:
            return
        try:
            await on_tool_call_started(fn_name, args)
        except Exception:
            pass
    cfg = load_app_config()
    provider, model = cfg.resolve_pair(provider_id, model_id)
    if provider is None or model is None:
        raise RuntimeError("no provider/model configured")
    if not provider.base_url:
        raise RuntimeError("provider base_url is empty")

    url, headers, base_payload = build_payload(
        provider=provider,
        model=model,
        messages=[],
        stream=False,
    )
    timeout_sec = provider_timeout_seconds(provider)

    messages: list[dict[str, Any]] = [{"role": m["role"], "content": m["content"]} for m in prompt_messages]
    sink = artifact_sink if artifact_sink is not None else []
    if tool_event_sink is None:
        # Build a throwaway queue so the rest of the body can use
        # ``emit`` uniformly without sprinkling None checks everywhere.
        # No consumer ever drains it, so events are silently dropped
        # (this matches the legacy behaviour where realtime_events
        # existed in memory but was never surfaced to the client).
        tool_event_sink = asyncio.Queue()

    def emit(name: str, data: dict[str, Any]) -> None:
        try:
            tool_event_sink.put_nowait((name, data))
        except Exception:
            pass

    inline_call_attempts: dict[str, int] = {}
    tool_executed = False
    last_tool_result_text = ""
    forced_retry_without_tool = False

    web_search_tool_schema = _pick_web_search_tool_schema(tools_schema)
    web_search_tool_name = _tool_name_from_schema(web_search_tool_schema) if web_search_tool_schema else None
    full_web_search_tool_schema = _find_tool_schema_by_name_fragment(tools_schema, "__web_search__full_web_search")
    full_web_search_tool_name = _tool_name_from_schema(full_web_search_tool_schema) if full_web_search_tool_schema else None
    last_user_query = (raw_user_query or _extract_last_user_query(messages)).strip()
    # The auto-search router already ran a query-rewriter before this
    # tool-loop was invoked.  When the model itself (or its fallback
    # path) decides to call the web-search tool, prefer the rewritten
    # query — otherwise we'd feed "Force web search" / "the artist"
    # verbatim to the search backend and get irrelevant results.
    last_user_query = (expanded_search_query or last_user_query).strip()
    provider_tools_disabled = False

    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        for _ in range(6):
            _, _, payload = build_payload(
                provider=provider,
                model=model,
                messages=messages,
                stream=False,
                tools=tools_schema if tools_schema and not provider_tools_disabled else None,
                tool_choice="auto" if tools_schema and not provider_tools_disabled else None,
            )

            try:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if provider_tools_disabled and tool_executed and last_tool_result_text:
                    return "Tool executed, but provider rejected the follow-up response request.\n" + last_tool_result_text
                if (
                    exc.response.status_code in {400, 404, 422}
                    and tools_schema
                    and not provider_tools_disabled
                    and not tool_executed
                    and web_search_tool_schema
                    and web_search_tool_name
                    and last_user_query
                ):
                    provider_tools_disabled = True
                    fallback_args = _build_search_fallback_args(web_search_tool_schema, last_user_query)
                    if not fallback_args:
                        raise
                    result = await execute_tool_call(web_search_tool_name, fallback_args)
                    _append_artifacts_from_tool_result(result, sink)
                    tool_executed = True
                    _save_message(
                        session_id=session_id,
                        window_id=window_id,
                        role="system",
                        content_text=f"Tool call {web_search_tool_name}: {json.dumps(fallback_args, ensure_ascii=False)}",
                        message_type="tool_call",
                        turn_id=turn_id,
                        source="tool",
                    )
                    tool_result_text = _tool_result_to_text(result)
                    compact_tool_text = _truncate_tool_text(tool_result_text)
                    last_tool_result_text = compact_tool_text
                    _save_message(
                        session_id=session_id,
                        window_id=window_id,
                        role="system",
                        content_text=f"Tool result {web_search_tool_name}:\n{compact_tool_text}",
                        message_type="tool_result",
                        turn_id=turn_id,
                        source="tool",
                    )
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "The provider rejected OpenAI tool schemas, so the orchestrator executed "
                                f"{web_search_tool_name} externally.\n\n"
                                f"Tool arguments:\n{json.dumps(fallback_args, ensure_ascii=False)}\n\n"
                                f"Tool result:\n{compact_tool_text}\n\n"
                                "Use this tool result as evidence and return the step output. "
                                "Do not claim that tools are unavailable."
                            ),
                        }
                    )
                    continue
                raise
            obj = resp.json()
            choices = obj.get("choices") or []
            message = (choices[0] if choices else {}).get("message") or {}

            content = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                inline_tool = _extract_inline_tool_call(str(content))
                if inline_tool:
                    fn_name, args = inline_tool
                    sig = f"{fn_name}|{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
                    inline_call_attempts[sig] = inline_call_attempts.get(sig, 0) + 1
                    if inline_call_attempts[sig] > 2:
                        if last_tool_result_text:
                            return (
                                "Tool executed, but model did not produce a final response. "
                                "Raw tool result:\n" + last_tool_result_text
                            )
                        return str(content).strip() or "No output from model."

                    result = await execute_tool_call(fn_name, args)
                    _append_artifacts_from_tool_result(result, sink)
                    tool_executed = True
                    _save_message(
                        session_id=session_id,
                        window_id=window_id,
                        role="system",
                        content_text=f"Tool call {fn_name}: {json.dumps(args, ensure_ascii=False)}",
                        message_type="tool_call",
                        turn_id=turn_id,
                        source="tool",
                    )
                    tool_result_text = (
                        _terminal_result_to_text(result) if fn_name == "run_terminal_command" else _tool_result_to_text(result)
                    )
                    compact_tool_text = _truncate_tool_text(tool_result_text)
                    last_tool_result_text = compact_tool_text
                    _save_message(
                        session_id=session_id,
                        window_id=window_id,
                        role="system",
                        content_text=f"Tool result {fn_name}:\n{compact_tool_text}",
                        message_type="tool_result",
                        turn_id=turn_id,
                        source="tool",
                    )

                    tc_id = str(uuid.uuid4())
                    messages.append({"role": "assistant", "content": str(content)})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": fn_name,
                            "content": compact_tool_text,
                        }
                    )
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Tool result is provided above. "
                                "Now return the final answer for the user directly. "
                                "Do not emit <tool_call> tags."
                            ),
                        }
                    )
                    continue

                if tools_schema and not tool_executed and not forced_retry_without_tool:
                    forced_retry_without_tool = True
                    messages.append({"role": "assistant", "content": str(content)})
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "You have tools available. "
                                "Call the most relevant tool now before responding to the user. "
                                "Do not provide placeholders and do not emit pseudo tags."
                            ),
                        }
                    )
                    continue

                if tools_schema and not tool_executed and web_search_tool_schema and web_search_tool_name and last_user_query:
                    fallback_args = _build_search_fallback_args(web_search_tool_schema, last_user_query)
                    if fallback_args:
                        result = await execute_tool_call(web_search_tool_name, fallback_args)
                        _append_artifacts_from_tool_result(result, sink)
                        tool_executed = True
                        _save_message(
                            session_id=session_id,
                            window_id=window_id,
                            role="system",
                            content_text=f"Tool call {web_search_tool_name}: {json.dumps(fallback_args, ensure_ascii=False)}",
                            message_type="tool_call",
                            turn_id=turn_id,
                            source="tool",
                        )
                        tool_result_text = _tool_result_to_text(result)
                        selected_tool_name = web_search_tool_name
                        selected_tool_text = tool_result_text

                        if (
                            _looks_like_empty_search_result(tool_result_text)
                            and full_web_search_tool_schema
                            and full_web_search_tool_name
                            and full_web_search_tool_name != web_search_tool_name
                        ):
                            secondary_args = _build_search_fallback_args(full_web_search_tool_schema, last_user_query)
                            if secondary_args:
                                secondary_result = await execute_tool_call(full_web_search_tool_name, secondary_args)
                                _append_artifacts_from_tool_result(secondary_result, sink)
                                _save_message(
                                    session_id=session_id,
                                    window_id=window_id,
                                    role="system",
                                    content_text=f"Tool call {full_web_search_tool_name}: {json.dumps(secondary_args, ensure_ascii=False)}",
                                    message_type="tool_call",
                                    turn_id=turn_id,
                                    source="tool",
                                )
                                secondary_text = _tool_result_to_text(secondary_result)
                                selected_tool_name = full_web_search_tool_name
                                selected_tool_text = secondary_text

                        compact_tool_text = _truncate_tool_text(selected_tool_text)
                        last_tool_result_text = compact_tool_text
                        _save_message(
                            session_id=session_id,
                            window_id=window_id,
                            role="system",
                            content_text=f"Tool result {selected_tool_name}:\n{compact_tool_text}",
                            message_type="tool_result",
                            turn_id=turn_id,
                            source="tool",
                        )

                        tc_id = str(uuid.uuid4())
                        messages.append({"role": "assistant", "content": str(content)})
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "name": selected_tool_name,
                                "content": compact_tool_text,
                            }
                        )
                        messages.append(
                            {
                                "role": "system",
                                "content": (
                                    "Tool result is provided above. "
                                    "Now return a concise final answer with concrete values. "
                                    "Do not use placeholders and do not emit <tool_call> tags."
                                ),
                            }
                        )
                        continue

                return str(content).strip() or "No output from model."

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
            )

            for tc in tool_calls:
                tc_id = tc.get("id") or str(uuid.uuid4())
                fn = tc.get("function") or {}
                fn_name = fn.get("name") or "unknown_tool"
                args = _safe_json_object(fn.get("arguments", "{}"))
                emit("tool_call_display", {"name": fn_name, "args": args})
                await _emit_tool_started(fn_name, args)
                result = await execute_tool_call(fn_name, args)
                _append_artifacts_from_tool_result(result, sink)
                tool_executed = True

                _save_message(
                    session_id=session_id,
                    window_id=window_id,
                    role="system",
                    content_text=f"Tool call {fn_name}: {json.dumps(args, ensure_ascii=False)}",
                    message_type="tool_call",
                    turn_id=turn_id,
                    source="tool",
                )
                tool_result_text = (
                    _terminal_result_to_text(result) if fn_name == "run_terminal_command" else _tool_result_to_text(result)
                )
                compact_tool_text = _truncate_tool_text(tool_result_text)
                last_tool_result_text = compact_tool_text
                _save_message(
                    session_id=session_id,
                    window_id=window_id,
                    role="system",
                    content_text=f"Tool result {fn_name}:\n{compact_tool_text}",
                    message_type="tool_result",
                    turn_id=turn_id,
                    source="tool",
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": fn_name,
                        "content": compact_tool_text,
                    }
                )

    if tool_executed and last_tool_result_text:
        return "Tool executed, but model did not produce a final response.\n" + last_tool_result_text
    return "Tool loop limit reached before final assistant response."



def _provider_stream_events(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Convert a single OpenAI-style streaming ``delta`` payload into a
    list of ``(event_name, text)`` tuples.

    The local model is asked to surface its reasoning under one of two
    field names (``reasoning_content`` on older checkpoints,
    ``reasoning`` on newer ones), and its visible answer under
    ``content``.  Empty / non-text values are dropped.
    """
    out: list[tuple[str, str]] = []
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return out
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        for field in ("reasoning_content", "reasoning"):
            value = delta.get(field)
            if isinstance(value, str) and value:
                out.append(("reasoning", value))
        content = delta.get("content")
        if isinstance(content, str) and content:
            out.append(("content", content))
    return out



def _detect_active_skill(user_content: str) -> str | None:
    """Extract an optional ``skill:<name>`` directive from the user
    message. This is the bridge between the chat UI and the SKILL.state
    runtime — when the user explicitly invokes a skill, the model sees
    only the (spec, state, observation) bundle and never replays the
    append-only conversation history."""
    import re

    match = re.search(r"\bskill\s*[:\-]\s*([a-z0-9_\-]+)", user_content, re.IGNORECASE)
    return match.group(1).lower() if match else None


def _auto_select_skill(user_content: str, *, threshold: int = 2) -> str | None:
    """Pick the best-matching skill for ``user_content`` using the same
    bag-of-words overlap that ``skills/executor.isSkillApplicable`` uses
    on the Node side. We re-implement it in Python here so the API
    never has to shell out into the Node registry.

    The match is intentionally conservative: a skill is only picked when
    at least ``threshold`` distinct tokens appear in both the user
    prompt and the skill's description / whenToUse fields. This stops
    generic prompts like "thanks" from accidentally activating a
    specific skill.
    """
    if not user_content:
        return None

    try:
        from .skill_state import list_skills_in_registry, _load_skill  # noqa: WPS433
    except Exception:
        return None

    candidates = list_skills_in_registry() or []
    if not candidates:
        return None

    prompt_tokens = {
        tok
        for tok in re.findall(r"[\w\u0400-\u04FF\u4E00-\u9FFF]+", user_content.lower())
        if len(tok) > 1
    }
    if not prompt_tokens:
        return None

    best_name: str | None = None
    best_score = 0
    for name in candidates:
        skill = _load_skill(name)
        if not isinstance(skill, dict):
            continue
        haystack = " ".join(
            [
                str(skill.get("description") or ""),
                str(skill.get("whenToUse") or ""),
            ]
        ).lower()
        skill_tokens = {
            tok
            for tok in re.findall(r"[\w\u0400-\u04FF\u4E00-\u9FFF]+", haystack)
            if len(tok) > 1
        }
        score = len(prompt_tokens & skill_tokens)
        if score > best_score:
            best_score = score
            best_name = name
    if best_score >= threshold and best_name is not None:
        return best_name
    return None


# Maximum number of consecutive identical assistant turns before we
# flag the session as ``stalled``. Set to 3 — that is the same
# threshold used by the upstream UI to surface "stuck" indicators.
STALLED_REPETITION_THRESHOLD = 3

# How similar two assistant turns must be (Jaccard overlap on word
# tokens) for us to count them as a repetition. Lowered from 0.85
# to 0.5 in the post-mortem: the local model rephrases the same
# clarification (e.g. "надайте список" → "уточніть, які саме
# сервіси") with pairwise overlap of only 0.3-0.4, so anything
# above 0.5 is a strong "the model is going in circles" signal.
STALLED_REPETITION_OVERLAP = 0.5

# How many of the last N assistant turns must independently look
# like a "clarification request" (short, re-asking the user) for the
# session to be flagged ``stalled``. The post-mortem showed the
# model rephrases the same clarification between turns, so a token
# overlap check alone is not enough — we look for the pattern of
# repeated clarification responses explicitly. Tied to the
# ``_is_clarification_request`` helper in ``memory.py``.
STALLED_CLARIFICATION_FRACTION = 1.0  # 100% of the last N must be clarifications


def _normalize_for_stall_check(text: str) -> str:
    """Lowercase, strip, collapse whitespace, drop punctuation. Used
    only for the Jaccard overlap calculation in
    :func:`_maybe_mark_session_stalled`."""
    if not text:
        return ""
    lowered = text.lower()
    # Strip non-letter / non-digit characters except whitespace.
    cleaned = re.sub(r"[^a-zа-яіїєґ0-9\s]+", " ", lowered, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def _jaccard_overlap(a: str, b: str) -> float:
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens or not b_tokens:
        return 0.0
    intersection = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return len(intersection) / max(1, len(union))


def _maybe_mark_session_stalled(session_id: str) -> None:
    """If the last ``STALLED_REPETITION_THRESHOLD`` assistant turns in
    the session are mutually very similar and there is no
    intervening user message, demote the session to
    ``status="stalled"`` so the UI can show the warning and the user
    can take action. Idempotent: only fires once per stall, and
    flips back to ``active`` as soon as a fresh user message lands.

    Two independent signals can trigger a stall:

    1. **Repetition**: the last N assistant turns have a pairwise
       Jaccard overlap above ``STALLED_REPETITION_OVERLAP``. Catches
       the case where the model literally repeats itself.

    2. **Repeated clarification**: the last N assistant turns each
       look like a clarification request (short, re-asking the user
       for the same input) — even if the phrasing varies slightly.
       Catches the case from the post-mortem where the local model
       rephrased the same "Будь ласка, надайте список сервісів…"
       question three times in a row with overlap ≈ 0.33.
    """
    from .memory import _is_clarification_request  # late import to avoid cycle

    # Pull the most recent assistant turns (up to a wider window so
    # the check sees ``STALLED_REPETITION_THRESHOLD`` of them even
    # when the user is sending clarification-acks between every
    # assistant reply).  We intentionally ignore the chat
    # interleaving: a "stalled" session is one in which the model
    # cannot make forward progress, regardless of what the user is
    # typing in the box.
    rows = fetch_all(
        """
        SELECT role, content_text, timestamp
        FROM messages
        WHERE session_id=? AND role='assistant'
        ORDER BY timestamp DESC
        LIMIT 12
        """,
        (session_id,),
    )
    if not rows:
        return
    # ``rows`` is descending; we want chronological order.
    rows.reverse()
    trailing = rows
    if len(trailing) < STALLED_REPETITION_THRESHOLD:
        return

    # --- Signal 1: pairwise Jaccard overlap ---
    normalised = [_normalize_for_stall_check(r["content_text"]) for r in trailing]
    base = normalised[-1]
    similar = 0
    if base:
        for text in normalised[-STALLED_REPETITION_THRESHOLD:]:
            if _jaccard_overlap(base, text) >= STALLED_REPETITION_OVERLAP:
                similar += 1
    repetition_match = similar >= STALLED_REPETITION_THRESHOLD

    # --- Signal 2: repeated clarification requests ---
    clarifications = 0
    for r in trailing[-STALLED_REPETITION_THRESHOLD:]:
        if _is_clarification_request(r["content_text"] or ""):
            clarifications += 1
    required_clarifications = max(
        2,
        int(round(STALLED_REPETITION_THRESHOLD * STALLED_CLARIFICATION_FRACTION)),
    )
    clarification_match = clarifications >= required_clarifications

    if not (repetition_match or clarification_match):
        return

    # Demote the session if it isn't already marked ``stalled``.
    current = fetch_one("SELECT status FROM sessions WHERE id=?", (session_id,))
    if current is None or current["status"] == "stalled":
        return
    execute(
        "UPDATE sessions SET status='stalled', updated_at=? WHERE id=?",
        (utcnow_iso(), session_id),
    )
    last_window = get_last_window(session_id)
    if last_window:
        detail = "repetition" if repetition_match else "repeated clarification"
        _save_message(
            session_id=session_id,
            window_id=last_window["id"],
            role="system",
            content_text=(
                f"Session marked as stalled ({detail}): the last "
                f"{STALLED_REPETITION_THRESHOLD} assistant turns did not progress. "
                "Send a fresh user message or restart the run to clear this state."
            ),
            message_type="internal_event",
            turn_id=f"stall-{uuid.uuid4()}",
            source="system",
        )


def _restore_stalled_session_if_needed(session_id: str) -> None:
    """Flip a ``stalled`` session back to ``active`` as soon as a
    fresh user message is processed. Keeps the UI list in sync
    with the actual state of the conversation."""
    row = fetch_one("SELECT status FROM sessions WHERE id=?", (session_id,))
    if row is None or row["status"] != "stalled":
        return
    execute(
        "UPDATE sessions SET status='active', updated_at=? WHERE id=?",
        (utcnow_iso(), session_id),
    )


async def stream_chat(
    session_id: str,
    user_content: str,
    thinking_mode_override: str | None = "session",
    *,
    provider_id: str | None = None,
    model_id: str | None = None,
    force_search: bool = False,
    bypass_search_cache: bool = False,
    active_skill: str | None = None,
    context_mode_override: str | None = None,
) -> AsyncIterator[str]:
    session_info = get_session(session_id)
    cfg = load_app_config()
    session_thinking_mode = session_info.get("thinking_mode", "medium")
    session_message_prefix_prompt = str(session_info.get("message_prefix_prompt") or "")
    effective_thinking_mode = (
        session_thinking_mode
        if thinking_mode_override in {None, "", "session"}
        else thinking_mode_override
    )
    # Per-turn override wins over the session default, which wins over
    # the global active pair.  ``None`` means "inherit" everywhere.
    resolved_provider_id = provider_id or session_info.get("provider_id")
    resolved_model_id = model_id or session_info.get("model_id")

    window = get_last_window(session_id)
    if not window:
        raise KeyError("window_not_found")
    window_id = window["id"]
    turn_id = str(uuid.uuid4())

    # Auto-restore a session that was previously flagged as
    # ``stalled``: the new user message is itself evidence the
    # user has taken action, so the loop is no longer active.
    _restore_stalled_session_if_needed(session_id)

    yield _sse("model_status", {"state": "thinking", "phase": "persist_user"})

    # Resolve the effective context mode. Per-turn override wins over
    # the session default which wins over the global AppConfig value.
    # This is the single source of truth that the rest of the turn
    # reads to decide whether to drop the chat history from the
    # prompt (SKILL.state) or keep it (full).
    effective_context_mode = (
        context_mode_override
        if context_mode_override in {"full", "skill_state"}
        else (session_info.get("context_mode") or cfg.context_mode or "full")
    )
    effective_context_mode = (
        effective_context_mode
        if effective_context_mode in {"full", "skill_state"}
        else "full"
    )
    yield _sse(
        "context_mode",
        {
            "mode": effective_context_mode,
            "auto": context_mode_override is None and not session_info.get("context_mode"),
        },
    )

    user_msg = _save_message(
        session_id=session_id,
        window_id=window_id,
        role="user",
        content_text=user_content,
        message_type="user",
        turn_id=turn_id,
    )

    maybe_update_durable_facts(session_id, user_content, source_message_id=user_msg["id"])

    # Pull the most recent user messages so the auto-search router can
    # expand short follow-ups ("the artist") into context-aware
    # queries ("tell me about Aposolix | the artist").  We
    # exclude the just-saved message — it's already in ``user_content``.
    # ``_recent_messages`` returns sqlite3.Row objects, so we use
    # bracket access (no ``.get`` on Row).
    _prior_user_rows = [
        m for m in _recent_messages(session_id, limit=10)
        if m["role"] == "user" and m["id"] != user_msg["id"]
    ]
    recent_user_messages = [str(m["content_text"] or "") for m in _prior_user_rows[-6:]]

    last_retrieval = fetch_one(
        "SELECT query_text FROM retrieval_logs WHERE session_id=? ORDER BY created_at DESC LIMIT 1",
        (session_id,),
    )
    retrieval_pack = None
    if _needs_retrieval(user_content, last_retrieval["query_text"] if last_retrieval else None):
        _save_message(
            session_id=session_id,
            window_id=window_id,
            role="system",
            content_text="Retrieval started.",
            message_type="internal_event",
            turn_id=turn_id,
            source="system",
        )
        yield _sse("retrieval_status", {"state": "running", "reason": "policy_trigger"})
        retrieval = run_retrieval(
            session_id=session_id,
            window_id=window_id,
            trigger_reason="policy_trigger",
            query=user_content,
            filters={"session_id": session_id},
        )
        retrieval_pack = retrieval["recall_pack"]
        _save_message(
            session_id=session_id,
            window_id=window_id,
            role="system",
            content_text=f"Retrieval completed. Log: {retrieval['log_id']}",
            message_type="internal_event",
            turn_id=turn_id,
            source="system",
        )
        yield _sse(
            "retrieval_status",
            {
                "state": "done",
                "log_id": retrieval["log_id"],
                "top": [r["chunk_id"] for r in retrieval["reranked_results"][:3]],
            },
        )

    # ------------------------------------------------------------------
    # Auto web search — the "google where I don't know" router.
    # Runs after the in-session retrieval and before the prompt is
    # assembled so the model sees grounded facts in the per-turn
    # section and can quote them instead of guessing.
    # ------------------------------------------------------------------
    auto_cfg = cfg.mcp_config.auto_search
    auto_decision = _should_auto_search(
        user_content,
        policy=auto_cfg.policy,
        enabled=auto_cfg.enabled,
        force=force_search,
        freshness_hints=auto_cfg.freshness_hints or None,
        factual_hints=auto_cfg.factual_hints or None,
        opinion_hints=auto_cfg.opinion_hints or None,
    )
    auto_result: AutoSearchResult | None = None
    if auto_decision.should_search:
        yield _sse(
            "auto_search",
            {
                "state": "running",
                "policy": auto_decision.policy,
                "reason": auto_decision.reason,
                "query": auto_decision.query,
            },
        )
        _save_message(
            session_id=session_id,
            window_id=window_id,
            role="system",
            content_text=(
                f"Auto-search triggered ({auto_decision.reason}, policy={auto_decision.policy})."
            ),
            message_type="auto_search_event",
            turn_id=turn_id,
            source="auto_search",
        )
        try:
            auto_result = await _run_auto_search(
                user_content,
                cfg=cfg,
                force=force_search,
                bypass_cache=bypass_search_cache,
                recent_user_messages=recent_user_messages,
                provider_id=resolved_provider_id,
                model_id=resolved_model_id,
            )
        except Exception as exc:
            auto_result = AutoSearchResult(
                query=auto_decision.query,
                normalized_query=auto_decision.normalized_query,
                answer="",
                citations=[],
                engine="",
                source="auto_search",
                cache_hit=False,
                took_ms=0,
                error=f"{type(exc).__name__}: {exc}",
            )
        # Cap the prompt-time footprint: the per-turn ``grounded_block``
        # never exceeds the configured max_chars, regardless of how
        # many citations the backend returned.
        if auto_result.grounded_block and auto_cfg.max_chars and len(auto_result.grounded_block) > auto_cfg.max_chars:
            auto_result.grounded_block = auto_result.grounded_block[: auto_cfg.max_chars].rstrip() + "\n…[truncated]"
        grounded_status = "ok" if (auto_result.citations and not auto_result.error) else (
            "cache_hit" if auto_result.cache_hit else "empty"
        )
        if auto_result.error:
            grounded_status = "error"
        yield _sse(
            "auto_search",
            {
                "state": "done",
                "status": grounded_status,
                "policy": auto_decision.policy,
                "reason": auto_decision.reason,
                "query": auto_result.query,
                "normalized_query": auto_result.normalized_query,
                "engine": auto_result.engine,
                "cache_hit": auto_result.cache_hit,
                "took_ms": int(auto_result.took_ms or 0),
                "answer_chars": len(auto_result.answer or ""),
                "citations": list(auto_result.citations or []),
                "error": auto_result.error or "",
            },
        )
        _save_message(
            session_id=session_id,
            window_id=window_id,
            role="system",
            content_text=(
                f"Auto-search {grounded_status} ({auto_result.took_ms}ms, "
                f"{len(auto_result.citations)} citation(s), engine={auto_result.engine or '-'}"
                f"{', cache_hit' if auto_result.cache_hit else ''}"
                f"{', error=' + auto_result.error if auto_result.error else ''})."
            ),
            message_type="auto_search_event",
            turn_id=turn_id,
            source="auto_search",
        )
        try:
            _record_auto_search_run(
                session_id=session_id,
                window_id=window_id,
                turn_id=turn_id,
                result=auto_result,
                policy=auto_decision.policy,
                trigger_reason=auto_decision.reason,
            )
        except Exception:
            # Telemetry is best-effort.
            pass
    elif force_search or auto_cfg.policy == "always" or auto_cfg.enabled:
        # The router considered the turn but declined; surface that to
        # the UI so the user can see why "Force web search" produced
        # no result for, e.g., an opinion prompt.
        yield _sse(
            "auto_search",
            {
                "state": "skipped",
                "policy": auto_decision.policy,
                "reason": auto_decision.reason,
                "query": auto_decision.query,
            },
        )

    terminal_tool_enabled = _needs_console_tool(user_content)
    file_tool_enabled = _needs_file_tool(user_content)
    user_wants_tools = _wants_tools(user_content) or terminal_tool_enabled or file_tool_enabled
    mcp_registry: MCPToolRegistry | None = None
    mcp_tools_schema: list[dict[str, Any]] = []
    prompt_tool_lines: list[str] = []

    mcp_servers = effective_mcp_servers(cfg)
    if user_wants_tools and cfg.mcp_config.enabled and mcp_servers:
        mcp_registry = await MCPToolRegistry.from_server_configs(mcp_servers)
        mcp_tools_schema = mcp_registry.tool_schemas()
        prompt_tool_lines.extend(mcp_registry.prompt_tool_lines())
        for err in mcp_registry.discovery_errors:
            yield _sse("tool_status", {"state": "error", "source": "mcp", "detail": f"{err['server']}: {err['error']}"})
            _save_message(
                session_id=session_id,
                window_id=window_id,
                role="system",
                content_text=f"MCP server unavailable: {err['server']} -> {err['error']}",
                message_type="internal_event",
                turn_id=turn_id,
                source="system",
            )

    if terminal_tool_enabled:
        prompt_tool_lines.append(_terminal_prompt_line())
    if file_tool_enabled:
        prompt_tool_lines.append(_file_tool_prompt_line())

    grounded_block = auto_result.grounded_block if auto_result and auto_result.grounded_block else None
    grounded_citations = list(auto_result.citations) if auto_result and auto_result.citations else None

    # SKILL.state integration: explicit `skill:<name>` directive in the
    # user message (or an override passed by the caller) flips the
    # prompt assembler into state-aware mode. We resolve the name,
    # initialise or resume the persisted state, and emit an SSE event
    # so the UI can show "Skill activated: …" without having to re-parse
    # the assistant's banner.
    #
    # Detection order:
    #   1. Caller-supplied `active_skill` (REST API override) — always
    #      wins, regardless of context_mode.
    #   2. Explicit ``skill:<name>`` directive in the user message —
    #      always wins, regardless of context_mode.
    #   3. Bag-of-words auto-routing — only fires when the user has
    #      opted into SKILL.state via ``context_mode == 'skill_state'``.
    #      In ``'full'`` mode the chat history stays in the prompt
    #      and the orchestrator never silently swaps it out.
    auto_select = effective_context_mode == "skill_state"
    detected_skill = (
        active_skill
        or _detect_active_skill(user_content)
        or (_auto_select_skill(user_content) if auto_select else None)
    )
    skill_bundle: dict[str, Any] | None = None
    if detected_skill:
        try:
            skill_state = start_or_resume_skill(
                session_id, detected_skill, user_prompt=user_content
            )
            skill_bundle = build_skill_prompt_bundle(session_id, detected_skill)
            yield _sse(
                "skill_state",
                {
                    "state": "activated",
                    "skill": detected_skill,
                    "current_step": skill_state.get("currentStep"),
                    "total_steps": skill_state.get("totalSteps"),
                    "status": skill_state.get("status"),
                    "auto": active_skill is None
                    and _detect_active_skill(user_content) is None,
                    "context_mode": effective_context_mode,
                },
            )
        except Exception as exc:
            yield _sse(
                "skill_state",
                {
                    "state": "error",
                    "skill": detected_skill,
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            )
            detected_skill = None

    prompt_messages = assemble_prompt(
        session_id,
        _with_message_prefix_prompt(user_content, session_message_prefix_prompt),
        cfg,
        retrieval_pack,
        thinking_mode=effective_thinking_mode,
        terminal_tool_enabled=terminal_tool_enabled,
        tool_instruction_lines=prompt_tool_lines or None,
        grounded_block=grounded_block,
        grounded_citations=grounded_citations,
        active_skill=detected_skill,
    )

    yield _sse("tool_status", {"state": "idle"})
    yield _sse("model_status", {"state": "streaming", "phase": "assistant_response"})
    assistant_chunks: list[str] = []
    provider_error_detail = ""
    tools_schema: list[dict[str, Any]] = []
    if terminal_tool_enabled:
        tools_schema.append(_terminal_tool_schema())
    if file_tool_enabled:
        tools_schema.append(_file_tool_schema())
    tools_schema.extend(mcp_tools_schema)
    turn_artifacts: list[dict[str, Any]] = []

    use_tool_path = terminal_tool_enabled or file_tool_enabled or (bool(mcp_tools_schema) and user_wants_tools)
    tool_event_sink: asyncio.Queue = asyncio.Queue()
    try:
        if use_tool_path and tools_schema:
            try:
                yield _sse("tool_status", {"state": "running"})

                # No more in-memory buffering: run the tool loop as a
                # background task and pump the tool_event_sink queue
                # concurrently.  This is what makes the chat UI feel
                # alive: each tool call, each reasoning delta, each
                # inline card reaches the browser the moment it
                # happens, in the correct order.
                loop_task = asyncio.create_task(
                    _run_with_tool_loop(
                        prompt_messages=prompt_messages,
                        tools_schema=tools_schema,
                        execute_tool_call=lambda fn_name, args: _execute_tool_call(
                            fn_name=fn_name,
                            args=args,
                            session_id=session_id,
                            session_info=session_info,
                            mcp_registry=mcp_registry,
                        ),
                        session_id=session_id,
                        window_id=window_id,
                        turn_id=turn_id,
                        raw_user_query=user_content,
                        expanded_search_query=(
                            auto_result.query if auto_result else ""
                        ),
                        artifact_sink=turn_artifacts,
                        tool_event_sink=tool_event_sink,
                        on_tool_call_started=None,
                        provider_id=resolved_provider_id,
                        model_id=resolved_model_id,
                    )
                )
                try:
                    while not loop_task.done():
                        try:
                            ev_name, ev_data = await asyncio.wait_for(
                                tool_event_sink.get(), timeout=0.25
                            )
                            yield _sse(ev_name, ev_data)
                        except asyncio.TimeoutError:
                            continue
                    while not tool_event_sink.empty():
                        ev_name, ev_data = tool_event_sink.get_nowait()
                        yield _sse(ev_name, ev_data)
                    tool_based_text = await loop_task
                except BaseException:
                    if not loop_task.done():
                        loop_task.cancel()
                    raise
                yield _sse("tool_status", {"state": "done"})
                for token in tool_based_text.split():
                    delta = token + " "
                    assistant_chunks.append(delta)
                    yield _sse("message_delta", {"delta": delta})
            except Exception as exc:
                # If tools path fails (provider may not support tools), fallback to stream mode.
                provider_error_detail = str(exc)
                yield _sse("tool_status", {"state": "error"})

        if not assistant_chunks:
            try:
                async for delta in _stream_from_provider(
                    prompt_messages,
                    provider_id=resolved_provider_id,
                    model_id=resolved_model_id,
                ):
                    assistant_chunks.append(delta)
                    yield _sse("message_delta", {"delta": delta})
            except Exception as exc:
                provider_error_detail = str(exc)
                _save_message(
                    session_id=session_id,
                    window_id=window_id,
                    role="system",
                    content_text=f"Provider error: {provider_error_detail}",
                    message_type="internal_event",
                    turn_id=turn_id,
                    source="system",
                )
                yield _sse(
                    "error",
                    {
                        "type": "provider_error",
                        "detail": provider_error_detail,
                    },
                )
                fallback = "Provider stream failed; switched to local fallback response."
                for token in fallback.split():
                    await asyncio.sleep(0.01)
                    delta = token + " "
                    assistant_chunks.append(delta)
                    yield _sse("message_delta", {"delta": delta})
    finally:
        if mcp_registry is not None:
            await mcp_registry.close()

    assistant_text = "".join(assistant_chunks).strip()
    if not assistant_text:
        assistant_text = "No output from model."

    assistant_msg = _save_message(
        session_id=session_id,
        window_id=window_id,
        role="assistant",
        content_text=assistant_text,
        message_type="assistant",
        turn_id=turn_id,
        content_json={"artifacts": turn_artifacts} if turn_artifacts else {},
    )

    # SKILL.state carry-over: extract concrete entities (service
    # names, URLs, bullet points) from the assistant turn we just
    # persisted and append them to ``durable_facts.json``.  The
    # next turn's prompt bundle reads that file, so the model
    # gets to see "previously found: LMSpeed, llm-stats, …" even
    # in SKILL.state mode where the chat history is dropped.
    try:
        # Also feed the grounded citations that fired during this
        # turn — they are the most reliable source of "what the
        # search found" because they are passed verbatim from the
        # search engine, regardless of how the model phrased its
        # answer.
        carryover_text = assistant_text
        if grounded_citations:
            for cite in grounded_citations:
                if not isinstance(cite, dict):
                    continue
                title = str(cite.get("title") or "").strip()
                url = str(cite.get("url") or "").strip()
                if title and url:
                    carryover_text += f"\n- {title} — {url}"
                elif url:
                    carryover_text += f"\n- {url}"
                elif title:
                    carryover_text += f"\n- {title}"
        record_turn_entities(
            session_id,
            carryover_text,
            source_message_id=assistant_msg.get("id"),
        )
    except Exception:
        # Memory carry-over is best-effort; a failure must not
        # break the turn.
        pass

    update_working_set(session_id)
    maybe_run_scheduled_wiki_lint(session_id)

    # Auto-detect a "stalled" loop: 3+ consecutive identical
    # assistant turns without an intervening user message. The last
    # session in this project hit this exact pattern (the model
    # asked "Будь ласка, надайте список сервісів…" three times in a
    # row). Demoting the session to ``stalled`` makes the UI list
    # mark it clearly so the user knows the assistant is no longer
    # progressing and can take action (re-prompt, restart run, etc.).
    _maybe_mark_session_stalled(session_id)

    token_limit, used_tokens, used_percent = _window_usage(session_id, window_id)
    pre_th = cfg.rollover_config.pre_rollover_threshold
    hard_th = cfg.rollover_config.hard_rollover_threshold

    if used_percent >= pre_th and window["pre_rollover_started_at"] is None:
        yield _sse(
            "rollover_status",
            {
                "state": "summarizing",
                "reason": "pre_rollover",
                "used_percent": round(used_percent, 4),
            },
        )
        _set_pre_rollover_started(window_id)
        cp = create_checkpoint(session_id, source_window_id=window_id, reason="pre_rollover")
        _save_message(
            session_id=session_id,
            window_id=window_id,
            role="system",
            content_text=f"Pre-rollover prepared. Checkpoint: {cp['id']}",
            message_type="internal_event",
            turn_id=turn_id,
            source="system",
        )
        yield _sse(
            "rollover_status",
            {
                "state": "prepared",
                "checkpoint_id": cp["id"],
                "used_percent": round(used_percent, 4),
            },
        )

    active_window_id = window_id
    if used_percent >= hard_th:
        yield _sse(
            "rollover_status",
            {
                "state": "summarizing",
                "reason": "hard_rollover",
                "used_percent": round(used_percent, 4),
            },
        )
        _set_hard_rollover_started(window_id)
        cp = create_checkpoint(session_id, source_window_id=window_id, reason="hard_rollover")
        new_window = create_next_window(session_id, closing_reason="token_limit", checkpoint_id=cp["id"])
        active_window_id = new_window["id"]
        _save_message(
            session_id=session_id,
            window_id=active_window_id,
            role="system",
            content_text=f"Hard rollover completed. New window: {active_window_id}",
            message_type="internal_event",
            turn_id=turn_id,
            source="system",
        )
        yield _sse(
            "rollover_status",
            {
                "state": "completed",
                "from_window": window_id,
                "to_window": active_window_id,
                "checkpoint_id": cp["id"],
            },
        )

    active_limit, active_used_tokens, active_used_percent = _window_usage(session_id, active_window_id)
    window_state = {
        "session_id": session_id,
        "window_id": active_window_id,
        "token_limit": active_limit,
        "used_tokens": active_used_tokens,
        "used_percent": round(active_used_percent, 4),
        "pre_rollover_threshold": pre_th,
        "hard_rollover_threshold": hard_th,
    }

    # Keep file-based session metadata in sync with DB state after each turn.
    write_session_json(session_id, get_session(session_id))

    yield _sse(
        "final_message",
        {
            "message": assistant_msg,
            "window_state": window_state,
            "user_message": user_msg,
            "artifacts": turn_artifacts,
            "auto_search": (auto_result.to_dict() if auto_result else None),
        },
    )
    yield _sse("model_status", {"state": "idle"})


def get_transcript(session_id: str) -> list[dict[str, Any]]:
    rows = fetch_all(
        "SELECT * FROM messages WHERE session_id=? ORDER BY timestamp ASC",
        (session_id,),
    )
    out = []
    for r in rows:
        item = dict(r)
        raw_content_json = item.get("content_json") or "{}"
        try:
            parsed_content_json = json.loads(raw_content_json) if isinstance(raw_content_json, str) else dict(raw_content_json)
        except Exception:
            parsed_content_json = {}
        artifacts = parsed_content_json.get("artifacts")
        if not isinstance(artifacts, list):
            artifacts = []
        artifacts = [a for a in artifacts if isinstance(a, dict)]
        item["content_json"] = parsed_content_json
        item["artifacts"] = artifacts
        item["is_pinned"] = bool(item["is_pinned"])
        item["is_anchor"] = bool(item["is_anchor"])
        out.append(item)
    return out


def get_window_state(session_id: str) -> dict[str, Any]:
    cfg = load_app_config()
    window = get_last_window(session_id)
    if not window:
        raise KeyError("window_not_found")
    token_limit, used_tokens, used_percent = _window_usage(session_id, window["id"])
    return {
        "session_id": session_id,
        "window_id": window["id"],
        "token_limit": token_limit,
        "used_tokens": used_tokens,
        "used_percent": round(used_percent, 4),
        "pre_rollover_threshold": cfg.rollover_config.pre_rollover_threshold,
        "hard_rollover_threshold": cfg.rollover_config.hard_rollover_threshold,
    }
