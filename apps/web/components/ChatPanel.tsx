'use client';

import { createPortal } from 'react-dom';
import { Dispatch, FormEvent, SetStateAction, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

import type {
  Artifact,
  AutoSearchCitation,
  AutoSearchResult,
  Message,
  ModelEntry,
  Provider,
  Run,
  ThinkingMode,
  ThinkingModeTurn,
  WindowState
} from '@/lib/api';
import {
  buildApiUrl,
  cancelRun,
  getTranscript,
  getWindowState,
  listProviders,
  listRuns,
  startBackgroundRun,
  streamChat,
  updateSession
} from '@/lib/api';

const AUTO_SCROLL_BOTTOM_GAP_PX = 24;
const MODEL_THINKING_PLACEHOLDER = 'Model is thinking…';
const SKILL_DISPLAY_PREFIX = '🔧'; // marker for skill display line

const THINKING_OPTIONS: ThinkingMode[] = ['off', 'low', 'medium', 'high'];

function normalizeAssistantMarkdown(input: string): string {
  const text = (input || '').replace(/\r\n?/g, '\n');
  if (!text || text.includes('```')) {
    return text;
  }

  return text
    .replace(/<\|?\/?tool_call\|?>/gi, '')
    .replace(/^\s*`?call:[^\n`]+`?\s*$/gim, '')
    .replace(/(^|\n)\s*call:([^\n]+)/g, '$1`call:$2`')
    .replace(/\*\*\*\s*(#{1,6}\s+)/g, '\n\n$1')
    .replace(/([^\n])\s+(#{1,6}\s+)/g, '$1\n\n$2')
    .replace(/([^\n])\s+(\d+\.\s+)/g, '$1\n$2')
    .replace(/([^\n])\s+([-*]\s+)/g, '$1\n$2')
    .replace(/([^\n])\s+(\*\*[^*\n]{2,120}:\*\*)/g, '$1\n\n$2')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return '';
  }
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function extractArtifacts(message: Message): Artifact[] {
  if (Array.isArray(message.artifacts)) {
    return message.artifacts.filter((item): item is Artifact => !!item && typeof item.download_url === 'string');
  }
  const nested = message.content_json?.artifacts;
  if (Array.isArray(nested)) {
    return nested.filter((item): item is Artifact => !!item && typeof item.download_url === 'string');
  }
  return [];
}

function deriveToolCallName(name: string | undefined): string | null {
  if (!name) return null;
  // mcp__skills_mcp__weather_dnipro -> weather-dnipro
  const stripped = name.replace(/^mcp__skills_mcp__/, '').replace(/^mcp__/,'');
  return stripped.replace(/_/g, '-') || name;
}

function AutoSearchIndicator({
  status
}: {
  status: {
    state: 'running' | 'done' | 'skipped' | 'error';
    policy?: string;
    reason?: string;
    query?: string;
    engine?: string;
    cache_hit?: boolean;
    took_ms?: number;
    answer_chars?: number;
    citations?: AutoSearchCitation[];
    error?: string;
  };
}) {
  if (status.state === 'running') {
    return <span className="chat-mini-status">🔎 Auto-searching…</span>;
  }
  if (status.state === 'skipped') {
    return (
      <span className="chat-mini-status" title={`Auto-search skipped (${status.reason || 'no signal'})`}>
        🔎 Auto-search skipped ({status.reason || 'no signal'})
      </span>
    );
  }
  if (status.state === 'error') {
    return (
      <span className="chat-mini-status" title={status.error || 'auto-search error'}>
        🔎 Auto-search error
      </span>
    );
  }
  const sources = status.citations?.length || 0;
  const label = status.cache_hit ? 'cache hit' : 'fresh';
  return (
    <span
      className="chat-mini-status"
      title={
        sources > 0
          ? `Auto-search: ${sources} source(s) (${label}, ${status.took_ms || 0}ms)`
          : 'Auto-search returned no sources'
      }
    >
      🔎 Auto-search: {sources} source{sources === 1 ? '' : 's'} ({label})
    </span>
  );
}

type Props = {
  sessionId: string | null;
  messages: Message[];
  onMessages: Dispatch<SetStateAction<Message[]>>;
  onModelState: (state: string) => void;
  onRetrievalState: (state: string) => void;
  onRolloverState: (state: string) => void;
  onWindowState: (state: WindowState) => void;
  sessionThinkingMode: ThinkingMode;
  sessionProviderId?: string | null;
  sessionModelId?: string | null;
  sessionHideSystemMessages?: boolean;
  sessionRunInBackground?: boolean;
  sessionForceSearchNext?: boolean;
  sessionBypassSearchCacheNext?: boolean;
  sessionContextMode?: 'full' | 'skill_state';
  onSessionUpdated?: (updated: {
    provider_id?: string | null;
    model_id?: string | null;
    thinking_mode?: ThinkingMode;
    hide_system_messages?: boolean;
    run_in_background?: boolean;
    force_search_next?: boolean;
    bypass_search_cache_next?: boolean;
  }) => void;
};

export default function ChatPanel({
  sessionId,
  messages,
  onMessages,
  onModelState,
  onRetrievalState,
  onRolloverState,
  onWindowState,
  sessionThinkingMode,
  sessionProviderId,
  sessionModelId,
  sessionHideSystemMessages,
  sessionRunInBackground,
  sessionForceSearchNext,
  sessionBypassSearchCacheNext,
  sessionContextMode,
  onSessionUpdated
}: Props) {
  const [draft, setDraft] = useState('');
  const [sending, setSending] = useState(false);
  // ``thinkingMode`` lives on the session itself so it survives reload
  // and is consistent with how the rest of the per-session settings
  // (provider/model, prefix prompt) are persisted.
  const [thinkingMode, setThinkingMode] = useState<ThinkingMode>(sessionThinkingMode);
  // Chat-header checkboxes: seeded from the session row so the chat
  // restores its toggle state when the user switches sessions.  The
  // individual ``onChange`` handlers persist the new value back via
  // ``updateSession`` so each session keeps its own preferences.
  const [hideSystemMessages, setHideSystemMessages] = useState<boolean>(Boolean(sessionHideSystemMessages));
  const [runInBackground, setRunInBackground] = useState<boolean>(Boolean(sessionRunInBackground));
  const [runs, setRuns] = useState<Run[]>([]);
  const [summaryInProgress, setSummaryInProgress] = useState(false);
  const [providers, setProviders] = useState<Provider[]>([]);
  const [activeProviderId, setActiveProviderId] = useState<string | null>(null);
  const [activeModelId, setActiveModelId] = useState<string | null>(null);
  // Per-session overrides.  ``undefined`` means "inherit" (session
  // value, then global active pair).  ``null`` means "explicitly
  // cleared".
  const [overrideProviderId, setOverrideProviderId] = useState<string | null | undefined>(undefined);
  const [overrideModelId, setOverrideModelId] = useState<string | null | undefined>(undefined);
  const [modelPickerOpen, setModelPickerOpen] = useState(false);
  // Persistent "force the auto-search router" toggle. Stays on between
  // turns so the user doesn't have to re-enable it for every message.
  const [forceSearchNext, setForceSearchNext] = useState<boolean>(Boolean(sessionForceSearchNext));
  // Persistent cache bypass — same lifetime as forceSearchNext.
  const [bypassSearchCacheNext, setBypassSearchCacheNext] = useState<boolean>(Boolean(sessionBypassSearchCacheNext));
  // Last auto-search event we received from the orchestrator — surfaced
  // in the chat controls so the user can see "searching…" and the
  // final citation list without expanding system messages.
  const [autoSearchStatus, setAutoSearchStatus] = useState<
    | {
        state: 'running' | 'done' | 'skipped' | 'error';
        policy?: string;
        reason?: string;
        query?: string;
        engine?: string;
        cache_hit?: boolean;
        took_ms?: number;
        answer_chars?: number;
        citations?: AutoSearchCitation[];
        error?: string;
      }
    | null
  >(null);
  const timelineRef = useRef<HTMLDivElement | null>(null);
  const isAtBottomRef = useRef(true);
  const wasBackgroundActiveRef = useRef(false);
  const modelPickerRef = useRef<HTMLDivElement | null>(null);
  const modelTriggerRef = useRef<HTMLButtonElement | null>(null);
  const [popoverPos, setPopoverPos] = useState<{ top: number; right: number; width: number } | null>(null);

  // Sync local override state and thinking mode whenever the session
  // itself changes.
  useEffect(() => {
    setOverrideProviderId(undefined);
    setOverrideModelId(undefined);
    setThinkingMode(sessionThinkingMode);
  }, [sessionId, sessionProviderId, sessionModelId, sessionThinkingMode]);

  // ``mounted`` flips to true on the first client render.  We use it
  // to delay rendering locale-sensitive strings (``toLocaleTimeString``
  // on message timestamps) until the client knows the user's timezone,
  // otherwise the SSR output and the first client render disagree and
  // React throws a hydration error.
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);

  // Restore the chat-header checkboxes from the session row whenever
  // the user switches sessions.
  useEffect(() => {
    setHideSystemMessages(Boolean(sessionHideSystemMessages));
    setRunInBackground(Boolean(sessionRunInBackground));
    setForceSearchNext(Boolean(sessionForceSearchNext));
    setBypassSearchCacheNext(Boolean(sessionBypassSearchCacheNext));
  }, [sessionId, sessionHideSystemMessages, sessionRunInBackground, sessionForceSearchNext, sessionBypassSearchCacheNext]);

  // When switching sessions, sync the chat-header checkboxes back to the
  // stored session defaults. ``sessionContextMode`` is locked at creation
  // time, so no equivalent setter exists — the label is read-only.
  useEffect(() => {
    setForceSearchNext(Boolean(sessionForceSearchNext));
    setBypassSearchCacheNext(Boolean(sessionBypassSearchCacheNext));
  }, [sessionId, sessionForceSearchNext, sessionBypassSearchCacheNext]);

  // Refresh the provider catalog once on mount.  We deliberately don't
  // refetch on every change so the popover stays snappy.
  useEffect(() => {
    let cancelled = false;
    void listProviders().then((resp) => {
      if (cancelled) return;
      setProviders(resp.providers || []);
      setActiveProviderId(resp.active_provider_id || null);
      setActiveModelId(resp.active_model_id || null);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Close the popover on outside click / Escape.
  useEffect(() => {
    if (!modelPickerOpen) return;
    function onDown(event: MouseEvent) {
      const target = event.target as Node;
      // The trigger button lives inside the picker container; the
      // popover itself is rendered via portal outside of it, so we
      // check both explicitly.
      if (
        modelPickerRef.current &&
        !modelPickerRef.current.contains(target) &&
        !popoverContentRef.current?.contains(target)
      ) {
        setModelPickerOpen(false);
      }
    }
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') setModelPickerOpen(false);
    }
    window.addEventListener('mousedown', onDown);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('keydown', onKey);
    };
  }, [modelPickerOpen]);

  // The popover is rendered through a portal so it can escape any
  // ancestor ``overflow: hidden`` (which would otherwise clip it on top
  // of the chat header).  Recompute its screen position whenever the
  // trigger moves, the window resizes, or the open state flips.
  const popoverContentRef = useRef<HTMLDivElement | null>(null);
  useLayoutEffect(() => {
    if (!modelPickerOpen) {
      setPopoverPos(null);
      return;
    }
    function recompute() {
      const btn = modelTriggerRef.current;
      if (!btn) return;
      const rect = btn.getBoundingClientRect();
      // Anchor to the right edge of the trigger so the popover doesn't
      // jitter sideways while the user types.
      setPopoverPos({
        top: rect.top - 6,
        right: window.innerWidth - rect.right,
        width: Math.max(260, rect.width)
      });
    }
    recompute();
    window.addEventListener('resize', recompute);
    window.addEventListener('scroll', recompute, true);
    return () => {
      window.removeEventListener('resize', recompute);
      window.removeEventListener('scroll', recompute, true);
    };
  }, [modelPickerOpen]);

  const enabledProviders = useMemo(
    () => providers.filter((p) => p.enabled && p.models.some((m) => m.enabled)),
    [providers]
  );

  const resolvedProviderId =
    overrideProviderId !== undefined
      ? overrideProviderId
      : sessionProviderId ?? activeProviderId ?? null;

  const providerForList =
    resolvedProviderId
      ? providers.find((p) => p.id === resolvedProviderId)
      : providers.find((p) => p.id === activeProviderId) || providers[0];

  const availableModels: ModelEntry[] = useMemo(() => {
    const provider = providerForList;
    if (!provider) return [];
    return provider.models.filter((m) => m.enabled);
  }, [providerForList]);

  const resolvedModelId =
    overrideModelId !== undefined
      ? overrideModelId
      : sessionModelId ?? (providerForList?.id === activeProviderId ? activeModelId : null);

  const effectiveModelId = resolvedModelId ?? availableModels[0]?.id ?? null;
  const effectiveProviderId = providerForList?.id ?? null;

  const currentProviderName =
    providerForList?.name || (effectiveProviderId ? 'Custom' : 'No provider');
  const currentModelName = (() => {
    if (!effectiveModelId) return 'No model';
    const m = availableModels.find((item) => item.id === effectiveModelId);
    if (m) return m.display_name || m.name;
    const fromAny = providers.flatMap((p) => p.models).find((m) => m.id === effectiveModelId);
    return fromAny ? fromAny.display_name || fromAny.name : 'Unknown model';
  })();

  useEffect(() => {
    setSummaryInProgress(false);
    isAtBottomRef.current = true;
  }, [sessionId]);

  useEffect(() => {
    const container = timelineRef.current;
    if (!container) {
      return;
    }
    if (isAtBottomRef.current) {
      container.scrollTop = container.scrollHeight;
    }
  }, [messages, hideSystemMessages]);

  useEffect(() => {
    if (!sessionId) {
      setRuns([]);
      wasBackgroundActiveRef.current = false;
      return;
    }
    const sid = sessionId;

    let cancelled = false;
    async function syncBackgroundRuns() {
      try {
        const runs = await listRuns(sid);
        if (cancelled) {
          return;
        }
        setRuns(runs);
        const active = runs.filter((run) => run.status === 'queued' || run.status === 'running').length;
        if (!sending) {
          onModelState(active > 0 ? 'background_run' : 'idle');
        }
        const shouldRefreshTranscript = !sending && (active > 0 || wasBackgroundActiveRef.current);
        const [transcript, ws] = await Promise.all([
          shouldRefreshTranscript ? getTranscript(sid) : Promise.resolve(null),
          getWindowState(sid).catch(() => null)
        ]);
        if (cancelled) {
          return;
        }
        if (ws) {
          onWindowState(ws);
        }
        if (transcript) {
          onMessages(transcript);
        }
        wasBackgroundActiveRef.current = active > 0;
      } catch {
        // Ignore polling errors; next tick retries.
      }
    }

    void syncBackgroundRuns();
    const timer = window.setInterval(() => {
      void syncBackgroundRuns();
    }, 4000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [sessionId, onMessages, onModelState, sending]);

  function updateBottomState() {
    const container = timelineRef.current;
    if (!container) {
      return;
    }
    const distanceToBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
    isAtBottomRef.current = distanceToBottom <= AUTO_SCROLL_BOTTOM_GAP_PX;
  }

  async function persistSession(
    patch: Partial<{
      provider_id: string | null;
      model_id: string | null;
      thinking_mode: ThinkingMode;
      hide_system_messages: boolean;
      run_in_background: boolean;
      force_search_next: boolean;
      bypass_search_cache_next: boolean;
    }>
  ) {
    if (!sessionId) return;
    try {
      await updateSession(sessionId, patch);
      onSessionUpdated?.(patch);
    } catch {
      /* local-only fallback; the session row is updated optimistically above */
    }
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!sessionId || !draft.trim() || sending) {
      return;
    }

    const content = draft.trim();
    setDraft('');
    setSending(true);
    const isBackgroundRun = runInBackground;
    // Persistent toggles: "Force web search" and "Bypass cache" stay
    // on between messages until the user explicitly turns them off,
    // matching how every other chat-level toggle (e.g. "Run in
    // background") behaves.
    const forceSearch = forceSearchNext;
    const bypassSearchCache = bypassSearchCacheNext;
    setAutoSearchStatus({ state: 'running' });

    const tempUserId = `tmp-user-${Date.now()}`;
    const tempAssistantId = `tmp-${Date.now()}`;
    onMessages([
      ...messages,
      {
        id: tempUserId,
        role: 'user',
        content_text: content,
        timestamp: new Date().toISOString(),
        message_type: 'user',
        is_pinned: false,
        is_anchor: false
      },
      ...(isBackgroundRun
        ? []
        : [
            {
              id: tempAssistantId,
              role: 'assistant',
              content_text: '',
              timestamp: new Date().toISOString(),
              message_type: 'assistant',
              is_pinned: false,
              is_anchor: false
            }
          ])
    ]);

    try {
      if (isBackgroundRun) {
        const run = await startBackgroundRun(sessionId, content);
        const transcript = await getTranscript(sessionId);
        onMessages(transcript);
        setRuns((prev) => [run, ...prev.filter((item) => item.id !== run.id)]);
        onModelState('background_run');
        onMessages((prev) => [
          ...prev,
          {
            id: `run-start-${run.id}`,
            role: 'system',
            content_text: `[Run ${run.id}] Started in the background. You can close the page and come back later.`,
            timestamp: new Date().toISOString(),
            message_type: 'internal_event',
            is_pinned: false,
            is_anchor: false
          }
        ]);
        return;
      }

      let accumulatedReasoning = '';
      await streamChat(
        sessionId,
        content,
        // The per-turn override of "session default" only applies when
        // the user actually picked a non-session value from the
        // popover; otherwise we send the session-level value so the
        // backend can decide whether to honour the session default.
        thinkingMode,
        ({ event, data }) => {
          if (event === 'message_delta') {
            onMessages((prev) =>
              prev.map((m) =>
                m.id === tempAssistantId
                  ? {
                      ...m,
                      content_text:
                        m.content_text === MODEL_THINKING_PLACEHOLDER
                          ? data.delta || ''
                          : m.content_text + (data.delta || '')
                    }
                  : m
              )
            );
            return;
          }
          if (event === 'reasoning_delta') {
            accumulatedReasoning += data.delta || '';
            onMessages((prev) =>
              prev.map((m) =>
                m.id === tempAssistantId
                  ? {
                      ...m,
                      content_json: {
                        ...(m.content_json || {}),
                        reasoning_text: accumulatedReasoning
                      }
                    }
                  : m
              )
            );
            return;
          }
          if (event === 'reasoning_status' && data.state === 'streaming') {
            onMessages((prev) =>
              prev.map((m) =>
                m.id === tempAssistantId && !m.content_text
                  ? { ...m, content_text: MODEL_THINKING_PLACEHOLDER }
                  : m
              )
            );
            return;
          }
          if (event === 'context_mode') {
            // The orchestrator tells us which prompt-building mode it
            // resolved for this turn. We surface a tiny pill so the
            // user can see when SKILL.state replaced the chat history
            // with a (spec, state, observation) bundle.
            const mode = data && typeof data.mode === 'string' ? data.mode : 'full';
            onMessages((prev) => [
              ...prev,
              {
                id: `ctx-mode-${Date.now()}`,
                session_id: sessionId,
                window_id: '',
                turn_id: '',
                role: 'system',
                timestamp: new Date().toISOString(),
                content_text:
                  mode === 'skill_state'
                    ? '🛠️ Context mode: SKILL.state (drop chat history, use validated execution state)'
                    : '📚 Context mode: full session (replay chat history)',
                content_json: { context_mode: mode, auto: Boolean(data && data.auto) },
                token_count: 0,
                message_type: 'context_mode_event',
                source: 'orchestrator',
                is_pinned: false,
                is_anchor: false,
                artifacts: []
              }
            ]);
            return;
          }
          if (event === 'skill_state') {
            // SKILL.state runtime reports which skill it activated for
            // the current turn (and whether the choice was auto-routed
            // vs. requested explicitly).
            const skillName = (data && data.skill) || '';
            const state = (data && data.state) || 'activated';
            const auto = Boolean(data && data.auto);
            const stepInfo = data && typeof data.current_step === 'number' && typeof data.total_steps === 'number'
              ? ` [step ${data.current_step}/${data.total_steps}]`
              : '';
            const text =
              state === 'error'
                ? `❌ SKILL.state error: ${data && data.detail ? String(data.detail) : 'unknown'}`
                : auto
                  ? `🪄 Auto-routed to SKILL.state skill: ${skillName}${stepInfo}`
                  : `🛠️ SKILL.state skill: ${skillName}${stepInfo}`;
            onMessages((prev) => [
              ...prev,
              {
                id: `skill-state-${Date.now()}`,
                session_id: sessionId,
                window_id: '',
                turn_id: '',
                role: 'system',
                timestamp: new Date().toISOString(),
                content_text: text,
                content_json: { skill_state: data },
                token_count: 0,
                message_type: 'skill_state_event',
                source: 'orchestrator',
                is_pinned: false,
                is_anchor: false,
                artifacts: []
              }
            ]);
            return;
          }
          if (event === 'tool_call_display') {
            const toolName = (data && data.name) || '';
            const friendlyName = deriveToolCallName(toolName) || toolName;
            const argsStr = data && data.args ? JSON.stringify(data.args) : '';
            const displayLine = `${SKILL_DISPLAY_PREFIX} Using skill: ${friendlyName}${argsStr ? ' (' + argsStr + ')' : ''}`;
            onMessages((prev) => [
              ...prev,
              {
                id: `tool-call-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
                role: 'system',
                content_text: displayLine,
                timestamp: new Date().toISOString(),
                message_type: 'tool_call',
                is_pinned: false,
                is_anchor: false
              }
            ]);
            return;
          }
          if (event === 'model_status') {
            const state = data.state || 'idle';
            onModelState(state);
            if (state === 'idle') {
              setSummaryInProgress(false);
            }
            return;
          }
          if (event === 'retrieval_status') {
            onRetrievalState(data.state || 'idle');
            return;
          }
          if (event === 'auto_search') {
            const state = (data && data.state) || 'running';
            setAutoSearchStatus({
              state: state as 'running' | 'done' | 'skipped' | 'error',
              policy: data?.policy,
              reason: data?.reason,
              query: data?.query,
              engine: data?.engine,
              cache_hit: Boolean(data?.cache_hit),
              took_ms: typeof data?.took_ms === 'number' ? data.took_ms : undefined,
              answer_chars: typeof data?.answer_chars === 'number' ? data.answer_chars : undefined,
              citations: Array.isArray(data?.citations) ? data.citations : [],
              error: data?.error || ''
            });
            if (state === 'done' && data?.citations && data.citations.length > 0) {
              const citations: AutoSearchCitation[] = data.citations;
              const lines = citations
                .map((c, idx) => `  [${idx + 1}] ${c.title || c.url} — ${c.url}`)
                .join('\n');
              onMessages((prev) => [
                ...prev,
                {
                  id: `auto-search-${Date.now()}`,
                  role: 'system',
                  content_text: `🔎 Auto-search (${data.engine || 'engine'}${data.cache_hit ? ', cache hit' : ''}, ${data.took_ms || 0}ms, ${citations.length} source(s)):\n${lines}`,
                  timestamp: new Date().toISOString(),
                  message_type: 'auto_search_event',
                  is_pinned: false,
                  is_anchor: false
                }
              ]);
            }
            return;
          }
          if (event === 'rollover_status') {
            const state = data.state || 'idle';
            onRolloverState(state);
            setSummaryInProgress(state === 'summarizing');
            return;
          }
          if (event === 'final_message') {
            setSummaryInProgress(false);
            const finalMessage = data.message as Message;
            const finalUser = data.user_message as Message;
            const ws = data.window_state as WindowState;
            onWindowState(ws);
            onMessages((prev) =>
              prev.map((m) => {
                if (m.id === tempAssistantId) {
                  const merged: Message = {
                    ...m,
                    ...finalMessage,
                    content_json: {
                      ...(m.content_json || {}),
                      ...(finalMessage.content_json || {}),
                      reasoning_text: accumulatedReasoning || (m.content_json && m.content_json.reasoning_text)
                    }
                  };
                  return merged;
                }
                if (m.id === tempUserId) {
                  return { ...m, ...finalUser };
                }
                return m;
              })
            );
            // If the orchestrator attached a final auto_search summary,
            // mirror it into the timeline so we have the citations even
            // when the streaming ``auto_search`` event was missed.
            const finalAuto = data?.auto_search as AutoSearchResult | null | undefined;
            if (finalAuto && finalAuto.citations && finalAuto.citations.length > 0) {
              const citations: AutoSearchCitation[] = finalAuto.citations;
              const lines = citations
                .map((c, idx) => `  [${idx + 1}] ${c.title || c.url} — ${c.url}`)
                .join('\n');
              onMessages((prev) => {
                // Avoid duplicating if the streaming event already
                // emitted a card with the same query.
                if (prev.some((m) => m.message_type === 'auto_search_event' && m.content_text.includes(finalAuto.query || ''))) {
                  return prev;
                }
                return [
                  ...prev,
                  {
                    id: `auto-search-final-${Date.now()}`,
                    role: 'system',
                    content_text: `🔎 Auto-search (${finalAuto.engine || 'engine'}${finalAuto.cache_hit ? ', cache hit' : ''}, ${finalAuto.took_ms || 0}ms, ${citations.length} source(s)):\n${lines}`,
                    timestamp: new Date().toISOString(),
                    message_type: 'auto_search_event',
                    is_pinned: false,
                    is_anchor: false
                  }
                ];
              });
            }
            return;
          }
          if (event === 'error') {
            setSummaryInProgress(false);
            const detail = typeof data?.detail === 'string' ? data.detail : 'Unknown error';
            onMessages((prev) => [
              ...prev,
              {
                id: `sys-${Date.now()}`,
                role: 'system',
                content_text: `Provider error: ${detail}`,
                timestamp: new Date().toISOString(),
                message_type: 'internal_event',
                is_pinned: false,
                is_anchor: false
              }
            ]);
          }
        },
        // The session's context mode is locked at creation time — the
        // orchestrator picks it up from the session row, so we don't
        // pass a per-turn override here.
        { provider_id: effectiveProviderId, model_id: effectiveModelId, force_search: forceSearch, bypass_search_cache: bypassSearchCache }
      );
    } finally {
      setSending(false);
    }
  }

  const visibleMessages = hideSystemMessages ? messages.filter((m) => m.role !== 'system') : messages;
  const activeRunCount = runs.filter((run) => run.status === 'queued' || run.status === 'running').length;
  const recentRuns = runs.slice(0, 4);

  async function onCancelRun(runId: string) {
    if (!sessionId) {
      return;
    }
    const updated = await cancelRun(sessionId, runId);
    setRuns((prev) => prev.map((run) => (run.id === runId ? updated : run)));
    const transcript = await getTranscript(sessionId);
    onMessages(transcript);
  }

  function progressText(run: Run): string {
    const progress = run.progress_json || {};
    const phase = typeof progress.phase === 'string' ? progress.phase : run.status;
    const current = typeof progress.current_step === 'number' ? progress.current_step : null;
    const total = typeof progress.total_steps === 'number' ? progress.total_steps : null;
    const verifier = typeof progress.verification_status === 'string' ? `, verifier: ${progress.verification_status}` : '';
    if (current !== null && total !== null && total > 0) {
      return `${phase} (${current}/${total}${verifier})`;
    }
    return `${phase}${verifier}`;
  }

  return (
    <section className="panel chat-panel">
      <div className="chat-controls-row">
        <div className="chat-controls-inline">
          <label className="inline-checkbox chat-inline-control">
            <input
              type="checkbox"
              checked={hideSystemMessages}
              onChange={(e) => {
                const next = e.target.checked;
                setHideSystemMessages(next);
                void persistSession({ hide_system_messages: next });
              }}
            />
            Hide system messages
          </label>
          <label className="inline-checkbox chat-inline-control">
            <input
              type="checkbox"
              checked={runInBackground}
              onChange={(e) => {
                const next = e.target.checked;
                setRunInBackground(next);
                void persistSession({ run_in_background: next });
              }}
            />
            Run in background
          </label>
          <label
            className="inline-checkbox chat-inline-control"
            title="Force the auto-search router for the next message (ignores global policy)."
          >
            <input
              type="checkbox"
              checked={forceSearchNext}
              onChange={(e) => {
                const next = e.target.checked;
                setForceSearchNext(next);
                void persistSession({ force_search_next: next });
              }}
            />
            Force web search
          </label>
          <label
            className="inline-checkbox chat-inline-control"
            title="Bypass the local auto-search cache for the next message."
          >
            <input
              type="checkbox"
              checked={bypassSearchCacheNext}
              onChange={(e) => {
                const next = e.target.checked;
                setBypassSearchCacheNext(next);
                void persistSession({ bypass_search_cache_next: next });
              }}
            />
            Bypass cache
          </label>
        </div>
        <div className="chat-controls-inline">
          {activeRunCount > 0 ? <span className="chat-mini-status">Background runs: {activeRunCount}</span> : null}
          {summaryInProgress ? <span className="chat-mini-status">Summarising context...</span> : null}
          {autoSearchStatus ? <AutoSearchIndicator status={autoSearchStatus} /> : null}
        </div>
      </div>
      {recentRuns.length > 0 ? (
        <div className="run-strip">
          {recentRuns.map((run) => {
            const isActive = run.status === 'queued' || run.status === 'running';
            return (
              <div key={run.id} className={`run-chip ${run.status}`}>
                <div className="run-chip-main">
                  <strong>{run.status}</strong>
                  <span>{progressText(run)}</span>
                </div>
                <small title={run.task_text}>{run.task_text}</small>
                {run.error_text ? <em>{run.error_text}</em> : null}
                {isActive ? (
                  <button type="button" onClick={() => void onCancelRun(run.id)}>
                    Cancel
                  </button>
                ) : null}
              </div>
            );
          })}
        </div>
      ) : null}
      <div className="chat-timeline" ref={timelineRef} onScroll={updateBottomState}>
        {visibleMessages.length === 0 ? (
          <div className="chat-timeline-empty">
            <span className="chat-timeline-empty-icon" aria-hidden>💬</span>
            <p className="chat-timeline-empty-title">
              {sessionId ? 'No messages yet' : 'Select or create a session to start chatting'}
            </p>
            <p className="chat-timeline-empty-hint">
              {sessionId
                ? 'Type a message below to begin the conversation.'
                : 'Pick a session from the list on the left, or create a new one.'}
            </p>
          </div>
        ) : null}
        {visibleMessages.map((m) => {
          const artifacts = extractArtifacts(m);
          return (
            <article key={m.id} className={`msg ${m.role}`}>
              <header>
                <strong>{m.role}</strong>
                <small suppressHydrationWarning>
                  {mounted ? new Date(m.timestamp).toLocaleTimeString() : ''}
                </small>
              </header>
              <div className="msg-content">
                {m.role === 'assistant' || m.role === 'system' ? (
                  <div className="msg-body">
                    {m.content_text === MODEL_THINKING_PLACEHOLDER || (m.content_json as any)?.reasoning_text ? (
                      <details className="msg-reasoning">
                        <summary>🧠 Agent reasoning</summary>
                        <div className="msg-reasoning-body">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {((m.content_json as any)?.reasoning_text) || 'Agent is thinking...'}
                          </ReactMarkdown>
                        </div>
                      </details>
                    ) : null}
                    {m.message_type === 'tool_call' ? (
                      <div className="msg-tool-call">{m.content_text}</div>
                    ) : m.message_type === 'auto_search_event' ? (
                      <pre className="msg-auto-search">{m.content_text}</pre>
                    ) : (
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {m.role === 'assistant' ? normalizeAssistantMarkdown(((m.content_text || '').replace(/<think>[\s\S]*?<\/think>/g, '').replace(/<think>[\s\S]*$/g, '').trim() || '...')) : m.content_text || '...'}
                      </ReactMarkdown>
                    )}
                  </div>
                ) : (
                  <p>{m.content_text || '...'}</p>
                )}
              </div>
              {m.role === 'assistant' && artifacts.length > 0 ? (
                <div className="msg-artifacts">
                  <strong>Files</strong>
                  <ul>
                    {artifacts.map((artifact) => (
                      <li key={artifact.id}>
                        <a href={buildApiUrl(artifact.download_url)} download={artifact.file_name}>
                          Download {artifact.file_name}
                        </a>
                        <small>{formatBytes(artifact.size_bytes)}</small>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </article>
          );
        })}
      </div>
      <form onSubmit={onSubmit} className="chat-input">
        <input
          className="chat-input-field"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Type your message"
        />
        {enabledProviders.length > 0 ? (
          <div className="chat-model-picker" ref={modelPickerRef}>
            <button
              type="button"
              ref={modelTriggerRef}
              className="chat-model-trigger"
              onClick={() => setModelPickerOpen((v) => !v)}
              title="Choose provider, model and thinking depth"
            >
              <span className="chat-model-trigger-label">Model</span>
              <span className="chat-model-trigger-summary">
                <span>{currentProviderName}</span>
                <span className="chat-model-trigger-sep">·</span>
                <span>{currentModelName}</span>
                <span className="chat-model-trigger-sep">·</span>
                <span>thinking: {thinkingMode}</span>
              </span>
              <span className="chat-model-trigger-caret" aria-hidden>▾</span>
            </button>
          </div>
        ) : null}
        <button className="chat-send" disabled={!sessionId || sending}>
          {sending ? '…' : 'Send'}
        </button>
      </form>
      {modelPickerOpen && popoverPos && enabledProviders.length > 0
        ? createPortal(
            <div
              ref={popoverContentRef}
              className="chat-model-popover"
              role="dialog"
              style={{
                position: 'fixed',
                top: popoverPos.top,
                right: popoverPos.right,
                width: popoverPos.width,
                transform: 'translateY(-100%)'
              }}
              onMouseDown={(e) => e.stopPropagation()}
            >
              <label className="chat-popover-field">
                <span>Provider</span>
                <select
                  value={effectiveProviderId ?? ''}
                  onChange={(e) => {
                    const next = e.target.value || null;
                    setOverrideProviderId(next);
                    setOverrideModelId(null);
                    void persistSession({ provider_id: next, model_id: null });
                  }}
                >
                  {enabledProviders.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="chat-popover-field">
                <span>Model</span>
                <select
                  value={effectiveModelId ?? ''}
                  onChange={(e) => {
                    const next = e.target.value || null;
                    setOverrideModelId(next);
                    void persistSession({
                      provider_id: effectiveProviderId,
                      model_id: next
                    });
                  }}
                >
                  {availableModels.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.display_name || m.name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="chat-popover-field">
                <span>Thinking</span>
                <select
                  value={thinkingMode}
                  onChange={(e) => {
                    const next = e.target.value as ThinkingMode;
                    setThinkingMode(next);
                    void persistSession({ thinking_mode: next });
                  }}
                >
                  {THINKING_OPTIONS.map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
                  ))}
                </select>
              </label>
              <p className="chat-popover-hint">
                Per-turn settings are remembered for this session.
              </p>
            </div>,
            document.body
          )
        : null}
    </section>
  );
}