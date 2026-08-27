// meta-skill-creator.js — Meta-skill for creating new skills

const { createSkill, getSkill } = require('./registry');
const { displaySkillCreated } = require('./chat-display');

/**
 * Analyse a user request and create a new skill from it.
 */
function createSkillFromRequest(userRequest) {
  const skillSpec = parseUserRequest(userRequest);

  if (!skillSpec) {
    return {
      success: false,
      message: 'Could not extract the skill specification from the request.',
    };
  }

  const existing = getSkill(skillSpec.name);
  if (existing) {
    return {
      success: false,
      message: `Skill "${skillSpec.name}" already exists. Use updateSkill to modify it.`,
    };
  }

  const skill = createSkill(
    skillSpec.name,
    skillSpec.description,
    skillSpec.instructions,
    skillSpec.whenToUse,
    skillSpec.examples
  );

  return {
    success: true,
    message: displaySkillCreated(skill),
    skill,
  };
}

/**
 * Parse a user request and extract the skill parameters from it.
 *
 * Trigger phrases cover English and Ukrainian so users can describe a
 * skill in either language without leaving the chat.
 */
function parseUserRequest(userRequest) {
  if (!userRequest || typeof userRequest !== 'string') return null;

  const request = userRequest.toLowerCase();

  // Determine the skill name.
  // Triggers cover English, the major European languages (Spanish,
  // French, Portuguese, German, Italian, Polish, Dutch, Turkish), the
  // major Asian languages (Chinese, Japanese, Korean, Hindi, Arabic,
  // Vietnamese), and Ukrainian. Russian is intentionally excluded.
  const nameMatch = userRequest.match(
    /(?:name|create|new|build|add|nombre|crear|nuevo|nuevo_habilidad|construir|añadir|nom|créer|nouveau|construire|ajouter|nome|criar|novo|construir|adicionar|name|erstellen|neu|hinzufügen|nome|crea|nuovo|aggiungi|nazwa|utwórz|nowy|dodaj|naam|maak|nieuwe|toevoegen|ad|oluştur|yeni|ekle|tạo|mới|thêm|名前|作成|新しい|追加|이름|만들기|새로운|추가|名称|创建|新建|添加|नाम|बनाएं|नया|जोड़ें|اسم|إنشاء|جديد|إضافة|назви|створи|новий|іменуй)\s+(?:skill|скіл|habilidad|compétence|habilidade|fähigkeit|abilità|umiejętność|vaardigheid|beceri|kỹ năng|スキル|스킬|技能|技能|कौशल|مهارة|навичка)?\s*[:\-]?\s*["']?([a-z0-9\-_]+)["']?/i
  );
  let name = nameMatch ? nameMatch[1] : null;

  if (!name) {
    // Fall back to generating a name from the prompt's keywords.
    name = generateSkillName(userRequest);
  }

  // Determine the description.
  const descMatch = userRequest.match(
    /(?:description|purpose|that|which|for|descripción|propósito|que|cuál|description|objectif|que|qui|descrição|finalidade|que|qual|beschreibung|zweck|das|was|descrizione|scopo|che|cosa|opis|cel|który|co|beschrijving|doel|wat|die|açıklama|amaç|ne|hangisi|mô tả|mục đích|説明|目的|それは何|설명|목적|그것은|描述|目的|它是什么|विवरण|उद्देश्य|यह क्या है|الوصف|الغاية|ما هو|опис|що\s+робить|призначення)[:\-]?\s*["']?([^"'\n]+)["']?/i
  );
  let description = descMatch ? descMatch[1].trim() : extractDescription(userRequest);

  // Determine the instructions.
  const instructions = extractInstructions(userRequest);

  // Determine whenToUse.
  const whenToUse = extractWhenToUse(userRequest);

  // Determine examples.
  const examples = extractExamples(userRequest);

  // Detect delegation: "via X", "using X", "with the help of X", or
  // Ukrainian equivalents ("через X", "за допомогою X", "використовуючи X")
  // followed by a tool name.
  const delegates_to = extractDelegation(userRequest);

  if (!name || !description) return null;

  return {
    name: name.toLowerCase().replace(/\s+/g, '-'),
    description,
    instructions,
    whenToUse,
    examples,
    delegates_to,
  };
}

/**
 * Detect whether the user request describes a delegating skill (one that
 * wraps another MCP tool). Returns a `delegates_to` config object or
 * `null` if the skill is self-contained.
 *
 * Patterns recognised (case-insensitive):
 *   English: "via <tool>" / "using <tool>" / "with the help of <tool>"
 *   Ukrainian: "через <tool>" / "за допомогою <tool>" / "використовуючи <tool>"
 *   "<tool>-based"
 *
 * The detected <tool> is mapped to the canonical MCP tool name.
 *
 * NOTE: weather, web-search, search and retriever all funnel into the
 * bundled web-search MCP. The `default_args` for weather pre-fills the
 * query so the agent does not have to think about the phrasing.
 */
function extractDelegation(text) {
  if (!text) return null;
  const lower = text.toLowerCase();

  // Known tool keywords with their canonical tool names and arg shape.
  // Aliases cover English, the major European languages, the major
  // Asian languages, and Ukrainian. Russian is intentionally excluded.
  const KNOWN_TOOLS = {
    // English
    'web-search':         { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    'web_search':         { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    'websearch':          { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    'search':             { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    'wikipedia':          { tool: 'wikipedia',         args_from: ['query', 'title'] },
    'wiki':               { tool: 'wikipedia',         args_from: ['query', 'title'] },
    'calculator':         { tool: 'calculator',        args_from: ['expression', 'formula'] },
    'calc':               { tool: 'calculator',        args_from: ['expression', 'formula'] },
    'math':               { tool: 'calculator',        args_from: ['expression', 'formula'] },
    'weather':            { tool: 'mcp__native_web_search__get_web_search_summaries', args_from: ['query'], default_args: { query: 'current weather today' } },
    'filesystem':         { tool: 'filesystem',        args_from: ['path', 'operation'] },
    'file':               { tool: 'filesystem',        args_from: ['path', 'operation'] },
    'terminal':           { tool: 'terminal',          args_from: ['command'] },
    'shell':              { tool: 'terminal',          args_from: ['command'] },
    'command':            { tool: 'terminal',          args_from: ['command'] },
    'retriever':          { tool: 'retriever',         args_from: ['query'] },
    'retrieve':           { tool: 'retriever',         args_from: ['query'] },
    'http':               { tool: 'http',              args_from: ['url'] },
    'fetch':              { tool: 'http',              args_from: ['url'] },
    // Spanish
    'búsqueda':           { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    'buscar':             { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    'calculadora':        { tool: 'calculator',        args_from: ['expression', 'formula'] },
    'clima':              { tool: 'mcp__native_web_search__get_web_search_summaries', args_from: ['query'], default_args: { query: 'clima actual hoy' } },
    'archivo':            { tool: 'filesystem',        args_from: ['path', 'operation'] },
    // French
    'recherche':          { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    'chercher':           { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    'calculatrice':       { tool: 'calculator',        args_from: ['expression', 'formula'] },
    'météo':              { tool: 'mcp__native_web_search__get_web_search_summaries', args_from: ['query'], default_args: { query: 'météo actuelle aujourd\'hui' } },
    'fichier':            { tool: 'filesystem',        args_from: ['path', 'operation'] },
    // Portuguese
    'pesquisa':           { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    'pesquisar':          { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    'calculadora':        { tool: 'calculator',        args_from: ['expression', 'formula'] },
    'tempo':              { tool: 'mcp__native_web_search__get_web_search_summaries', args_from: ['query'], default_args: { query: 'tempo atual hoje' } },
    'arquivo':            { tool: 'filesystem',        args_from: ['path', 'operation'] },
    // German
    'suche':              { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    'suchen':             { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    'taschenrechner':     { tool: 'calculator',        args_from: ['expression', 'formula'] },
    'wetter':             { tool: 'mcp__native_web_search__get_web_search_summaries', args_from: ['query'], default_args: { query: 'aktuelles Wetter heute' } },
    'datei':              { tool: 'filesystem',        args_from: ['path', 'operation'] },
    // Italian
    'ricerca':            { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    'cercare':            { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    'calcolatrice':       { tool: 'calculator',        args_from: ['expression', 'formula'] },
    'meteo':              { tool: 'mcp__native_web_search__get_web_search_summaries', args_from: ['query'], default_args: { query: 'meteo attuale oggi' } },
    // Polish
    'wyszukiwanie':       { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    'szukaj':             { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    'kalkulator':         { tool: 'calculator',        args_from: ['expression', 'formula'] },
    'pogoda':             { tool: 'mcp__native_web_search__get_web_search_summaries', args_from: ['query'], default_args: { query: 'aktualna pogoda dzisiaj' } },
    // Dutch
    'zoeken':             { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    'rekenmachine':       { tool: 'calculator',        args_from: ['expression', 'formula'] },
    // Turkish
    'arama':              { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    'hesap':              { tool: 'calculator',        args_from: ['expression', 'formula'] },
    'hava':               { tool: 'mcp__native_web_search__get_web_search_summaries', args_from: ['query'], default_args: { query: 'bugünkü hava durumu' } },
    // Vietnamese
    'tìm':                { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    'máy tính':           { tool: 'calculator',        args_from: ['expression', 'formula'] },
    'thời tiết':          { tool: 'mcp__native_web_search__get_web_search_summaries', args_from: ['query'], default_args: { query: 'thời tiết hôm nay' } },
    // Japanese
    '検索':                { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    '電卓':                { tool: 'calculator',        args_from: ['expression', 'formula'] },
    '天気':                { tool: 'mcp__native_web_search__get_web_search_summaries', args_from: ['query'], default_args: { query: '今日の天気' } },
    // Korean
    '검색':                { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    '계산기':              { tool: 'calculator',        args_from: ['expression', 'formula'] },
    '날씨':                { tool: 'mcp__native_web_search__get_web_search_summaries', args_from: ['query'], default_args: { query: '오늘 날씨' } },
    // Chinese
    '搜索':                { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    '计算器':              { tool: 'calculator',        args_from: ['expression', 'formula'] },
    '天气':                { tool: 'mcp__native_web_search__get_web_search_summaries', args_from: ['query'], default_args: { query: '今天的天气' } },
    // Hindi
    'खोज':                { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    'कैलकुलेटर':          { tool: 'calculator',        args_from: ['expression', 'formula'] },
    'मौसम':                { tool: 'mcp__native_web_search__get_web_search_summaries', args_from: ['query'], default_args: { query: 'आज का मौसम' } },
    // Arabic
    'بحث':                { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    'آلة حاسبة':          { tool: 'calculator',        args_from: ['expression', 'formula'] },
    'الطقس':                { tool: 'mcp__native_web_search__get_web_search_summaries', args_from: ['query'], default_args: { query: 'طقس اليوم' } },
    // Ukrainian
    'пошук':              { tool: 'native-web-search', args_from: ['query', 'q', 'search_query'] },
    'калькулятор':        { tool: 'calculator',        args_from: ['expression', 'formula'] },
    'погода':             { tool: 'mcp__native_web_search__get_web_search_summaries', args_from: ['query'], default_args: { query: 'поточна погода сьогодні' } },
    'файл':               { tool: 'filesystem',        args_from: ['path', 'operation'] },
  };

  // Delegation trigger phrases.
  // Triggers cover the major languages plus Ukrainian; Russian is
  // intentionally excluded.
  const patterns = [
    /\b(?:via|using|using the|with the help of|usando|utilizando|con ayuda de|avec|aide de|en utilisant|utilizando|com ajuda|usando|mit hilfe von|verwenden|utilizzando|con l'aiuto|używając|pomocą|met behulp van|gebruik|kullanarak|yardımıyla|sử dụng|với sự giúp đỡ|使用|사용|사용하여|使用|使用|इस्तेमाल|के स помощью|باستخدام|مساعدة|через|за\s+допомогою|використовуючи)\s+([a-zа-яіїєґ\u00C0-\u017F\u0400-\u04FF][a-z0-9а-яіїєґ\u00C0-\u017F\u0400-\u04FF_-]*)/i,
    /\b([a-zа-яіїєґ\u00C0-\u017F\u0400-\u04FF][a-z0-9а-яіїєґ\u00C0-\u017F\u0400-\u04FF_-]*)-(?:based|tool|mcp)\b/i,
    /\b(uses?|using)\s+([a-z_][a-z0-9_-]+)/i,
  ];

  for (const pattern of patterns) {
    const m = text.match(pattern);
    if (!m) continue;
    // The tool keyword is the last capture group (most patterns
    // only have one; the "uses X" pattern has two - we want the tool).
    const toolWord = (m[m.length - 1] || '').toLowerCase();
    if (KNOWN_TOOLS[toolWord]) {
      return KNOWN_TOOLS[toolWord];
    }
    // Try matching substring: e.g. "web search" -> "web-search"
    const joined = toolWord.replace(/[\s_]+/g, '-');
    if (KNOWN_TOOLS[joined]) {
      return KNOWN_TOOLS[joined];
    }
  }

  return null;
}

/**
 * Generate a skill name from the prompt text.
 * The character class covers Latin, extended Latin (for Western
 * European languages), and Cyrillic letters (Ukrainian) so prompts
 * in those languages produce sensible name components.
 */
function generateSkillName(text) {
  const words = text
    .toLowerCase()
    .replace(/[^a-zа-яіїєґ\u00C0-\u017F\u4E00-\u9FFF0-9\s]/gi, '')
    .split(/\s+/)
    .filter((w) => w.length > 3)
    .slice(0, 3)
    .join('-');
  return words || 'new-skill';
}

/**
 * Extract a description from the prompt text.
 * Trigger words cover English, the major European and Asian languages,
 * and Ukrainian. Russian is intentionally excluded.
 */
function extractDescription(text) {
  const match = text.match(
    /(?:that|which|for|purpose|does|to|que|qué|cuál|qui|que|quoi|que|qual|das|was|für|che|cosa|który|co|wat|wat|wat|ne|hangisi|それは何|それは|それは何|그것은|그것|그것은|它是什么|它|那个|यह क्या है|यह|वह|ما هو|ما|ذلك|який|що|для|призначений\s+для|робить)\s+([^.!?\n]+)/i
  );
  if (match) return match[1].trim();

  // Otherwise fall back to the first sentence.
  const firstSentence = text.split(/[.!?\n]/)[0];
  return firstSentence.substring(0, 100).trim();
}

/**
 * Extract a list of instructions from the prompt text.
 */
function extractInstructions(text) {
  const instructions = [];

  // Look for numbered steps first.
  const numberedSteps = text.matchAll(/\d+[.)]\s+([^.!?\n]+)/g);
  for (const match of numberedSteps) {
    instructions.push(match[1].trim());
  }

  if (instructions.length > 0) return instructions;

  // Fall back to sentences containing step-like keywords.
  // Keywords cover English, the major European and Asian languages,
  // and Ukrainian. Russian is intentionally excluded.
  const keywords = [
    // English
    'step', 'first', 'then', 'next', 'finally', 'after that', 'before that',
    // Spanish
    'paso', 'primero', 'luego', 'después', 'finalmente',
    // French
    'étape', 'premièrement', 'ensuite', 'puis', 'enfin',
    // Portuguese
    'passo', 'primeiro', 'em seguida', 'depois', 'finalmente',
    // German
    'schritt', 'zuerst', 'dann', 'anschließend', 'schließlich',
    // Italian
    'passo', 'prima', 'quindi', 'poi', 'infine',
    // Polish
    'krok', 'najpierw', 'potem', 'następnie', 'na koniec',
    // Dutch
    'stap', 'eerst', 'vervolgens', 'daarna', 'tot slot',
    // Turkish
    'adım', 'önce', 'sonra', 'ardından', 'son olarak',
    // Vietnamese
    'bước', 'đầu tiên', 'sau đó', 'rồi', 'cuối cùng',
    // Japanese
    'ステップ', '最初', '次に', 'その後', '最後に',
    // Korean
    '단계', '먼저', '다음', '그 다음', '마지막으로',
    // Chinese
    '步骤', '首先', '然后', '接着', '最后',
    // Hindi
    'कदम', 'पहले', 'फिर', 'बाद में', 'अंत में',
    // Arabic
    'خطوة', 'أولا', 'ثم', 'بعد ذلك', 'أخيرا',
    // Ukrainian
    'крок', 'етап', 'спочатку', 'потім', 'далі', 'після цього',
  ];
  const sentences = text.split(/[.!?\n]+/).filter((s) => s.trim());

  for (const sentence of sentences) {
    const lower = sentence.toLowerCase();
    if (keywords.some((k) => lower.includes(k))) {
      instructions.push(sentence.trim());
    }
  }

  if (instructions.length === 0) {
    instructions.push('Execute the user request');
  }

  return instructions;
}

/**
 * Extract the `whenToUse` hint from the prompt.
 * Trigger phrases cover English, the major European and Asian
 * languages, and Ukrainian. Russian is intentionally excluded.
 */
function extractWhenToUse(text) {
  const match = text.match(
    /(?:when|use|uses|used|cuando|usar|usa|cuándo|utiliser|quand|utilisation|quando|usar|quando|wann|verwenden|uso|quando|utilizzare|usare|kiedy|używać|gebruiken|wanneer|ne zaman|kullanmak|khi nào|sử dụng|いつ|使用|언제|사용|何时|使用|कब|उपयोग|متى|استخدام|коли|використовувати|застосовувати)[:\-]?\s*([^.!?\n]+)/i
  );
  return match ? match[1].trim() : null;
}

/**
 * Extract example prompts from the text.
 * Trigger words cover English, the major European and Asian
 * languages, and Ukrainian. Russian is intentionally excluded.
 */
function extractExamples(text) {
  const examples = [];

  const exampleMatches = text.matchAll(
    /(?:example|e\.g\.?|ejemplo|ej\.|exemple|ex\.|exemplo|ex\.|beispiel|z\.b\.|esempio|es\.|przykład|np\.|voorbeeld|b\.v\.|örneğin|vd\.|ví dụ|例|例えば|예|예를 들어|例如|比如|उदाहरण|जैसे|مثال|приклад)[:\-]?\s*["']?([^"'\n]+)["']?/gi
  );
  for (const match of exampleMatches) {
    examples.push({
      prompt: match[1].trim(),
      action: 'Run the skill',
    });
  }

  return examples.length > 0 ? examples : undefined;
}

module.exports = { createSkillFromRequest, parseUserRequest, extractDelegation };