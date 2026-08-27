'use client';

import Link from 'next/link';
import { FormEvent, useEffect, useMemo, useState } from 'react';

import {
  activateProviderModel,
  addProviderModel,
  createProvider,
  deleteProvider,
  deleteProviderModel,
  discoverMcpTools,
  getSettings,
  listProviders,
  setActiveSelection,
  testAutoSearch,
  updateProvider,
  updateProviderModel,
  updateSettings,
  validateProvider,
  type AutoSearchCitation,
  type AutoSearchConfig,
  type AutoSearchResult,
  type AutoSearchTestResponse,
  type MCPDiscovery,
  type ModelEntry,
  type ModelEntryInput,
  type Provider,
  type ProviderCreateInput
} from '@/lib/api';

function normalizeBaseUrl(value: string): string {
  return value.trim().replace(/(https?:\/\/[^/]+)\/:(\d+)(\/?.*)$/i, '$1:$2$3');
}

function parseThreshold(value: string): number {
  const normalized = value.replace(',', '.').trim();
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed : 0;
}

function parsePositiveIntOrNull(value: string): number | null {
  const normalized = value.trim();
  if (!normalized) {
    return null;
  }
  const parsed = Number(normalized);
  if (!Number.isFinite(parsed)) {
    return null;
  }
  return Math.max(1, Math.floor(parsed));
}

function parseBoundedInt(value: string | number, fallback: number, min: number, max: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, Math.floor(parsed)));
}

function prettyJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

type ValidationState = {
  ok: boolean;
  detail: string;
  models: string[];
} | null;

type EditingProvider = {
  id?: string;
  name: string;
  provider_name: string;
  base_url: string;
  endpoint: string;
  api_key: string;
  request_timeout_sec: number;
  enabled: boolean;
  notes: string;
};

function emptyEditingProvider(): EditingProvider {
  return {
    name: '',
    provider_name: 'openai-compatible',
    base_url: '',
    endpoint: '/chat/completions',
    api_key: '',
    request_timeout_sec: 240,
    enabled: true,
    notes: ''
  };
}

// Friendly labels for the provider-type picker in the settings form.
// We only carry the raw ``provider_name`` (a string) on the wire so any
// new value works without a schema change.
const PROVIDER_KIND_OPTIONS: Array<{ value: string; label: string }> = [
  { value: 'openai-compatible', label: 'OpenAI-compatible' },
  { value: 'openai', label: 'OpenAI' },
  { value: 'anthropic', label: 'Anthropic' },
  { value: 'google', label: 'Google Gemini' },
  { value: 'ollama', label: 'Ollama' },
  { value: 'lm-studio', label: 'LM Studio' },
  { value: 'vllm', label: 'vLLM' },
  { value: 'custom', label: 'Custom' }
];

export default function SettingsPage() {
  const [config, setConfig] = useState<any>(null);
  const [providers, setProviders] = useState<Provider[]>([]);
  const [activeProviderId, setActiveProviderId] = useState<string | null>(null);
  const [activeModelId, setActiveModelId] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [providerSaving, setProviderSaving] = useState(false);
  const [modelSaving, setModelSaving] = useState<string | null>(null);
  const [validation, setValidation] = useState<ValidationState>(null);
  const [mcpServersText, setMcpServersText] = useState('[]');
  const [mcpParseError, setMcpParseError] = useState<string | null>(null);
  const [mcpDiscovery, setMcpDiscovery] = useState<MCPDiscovery | null>(null);
  const [editingProvider, setEditingProvider] = useState<EditingProvider | null>(null);
  const [editingProviderId, setEditingProviderId] = useState<string | null>(null);
  const [editingModelDraft, setEditingModelDraft] = useState<Record<string, ModelEntry>>({});
  const [newModelDraft, setNewModelDraft] = useState<Record<string, ModelEntryInput>>({});
  const [providerValidation, setProviderValidation] = useState<Record<string, ValidationState>>({});
  const [providerError, setProviderError] = useState<string | null>(null);
  // Per-provider success/info message shown under the model table so
  // the user gets positive feedback when "Add model" actually creates
  // a row (the previous implementation only surfaced *errors*, which
  // made it look like nothing happened on a quiet success).
  const [modelMessage, setModelMessage] = useState<Record<string, string>>({});
  const [autoSearchTestQuery, setAutoSearchTestQuery] = useState('');
  const [autoSearchTestForce, setAutoSearchTestForce] = useState(false);
  const [autoSearchTestBypass, setAutoSearchTestBypass] = useState(false);
  const [autoSearchTestResult, setAutoSearchTestResult] = useState<AutoSearchTestResponse | null>(null);
  const [autoSearchTestRunning, setAutoSearchTestRunning] = useState(false);
  const [autoSearchTestError, setAutoSearchTestError] = useState<string | null>(null);

  useEffect(() => {
    void getSettings().then((loaded) => {
      setConfig(loaded);
      setMcpServersText(prettyJson(loaded?.mcp_config?.servers || []));
    });
    void refreshProviders();
  }, []);

  async function refreshProviders() {
    try {
      const list = await listProviders();
      setProviders(list.providers || []);
      setActiveProviderId(list.active_provider_id || null);
      setActiveModelId(list.active_model_id || null);
    } catch (err) {
      setProviderError(err instanceof Error ? err.message : 'Failed to load providers');
    }
  }

  async function onSave(e: FormEvent) {
    e.preventDefault();
    if (!config || saving) {
      return;
    }

    let parsedServers: any[] = [];
    try {
      const parsed = JSON.parse(mcpServersText || '[]');
      if (!Array.isArray(parsed)) {
        throw new Error('MCP servers must be a JSON array');
      }
      parsedServers = parsed;
      setMcpParseError(null);
    } catch (err) {
      setMcpParseError(err instanceof Error ? err.message : 'Invalid JSON');
      return;
    }

    setSaving(true);
    try {
      const autoSearchConfig = config.mcp_config?.auto_search || {};
      const normalized = {
        ...config,
        mcp_config: {
          ...(config.mcp_config || {}),
          enabled: Boolean(config.mcp_config?.enabled),
          native_web_search_timeout_sec: parseBoundedInt(
            config.mcp_config?.native_web_search_timeout_sec ?? 45,
            45,
            3,
            600
          ),
          servers: parsedServers,
          auto_search: {
            enabled: Boolean(autoSearchConfig.enabled),
            policy: ['off', 'auto', 'always'].includes(autoSearchConfig.policy) ? autoSearchConfig.policy : 'auto',
            max_chars: parseBoundedInt(autoSearchConfig.max_chars ?? 4000, 4000, 0, 20000),
            cache_ttl_sec: parseBoundedInt(autoSearchConfig.cache_ttl_sec ?? 21600, 21600, 0, 30 * 24 * 3600),
            max_per_turn: parseBoundedInt(autoSearchConfig.max_per_turn ?? 1, 1, 0, 4),
            max_citations: parseBoundedInt(autoSearchConfig.max_citations ?? 5, 5, 1, 10),
            summary_max_chars: parseBoundedInt(autoSearchConfig.summary_max_chars ?? 1200, 1200, 0, 8000),
            snippet_per_source_chars: parseBoundedInt(
              autoSearchConfig.snippet_per_source_chars ?? 320,
              320,
              0,
              4000
            ),
            include_snippets: Boolean(autoSearchConfig.include_snippets ?? true),
            include_full_content: Boolean(autoSearchConfig.include_full_content ?? false),
            prefer_engine: String(autoSearchConfig.prefer_engine ?? ''),
            freshness_hints: Array.isArray(autoSearchConfig.freshness_hints)
              ? autoSearchConfig.freshness_hints.filter((s: any) => typeof s === 'string')
              : [],
            factual_hints: Array.isArray(autoSearchConfig.factual_hints)
              ? autoSearchConfig.factual_hints.filter((s: any) => typeof s === 'string')
              : [],
            opinion_hints: Array.isArray(autoSearchConfig.opinion_hints)
              ? autoSearchConfig.opinion_hints.filter((s: any) => typeof s === 'string')
              : []
          }
        }
      };
      const saved = await updateSettings(normalized);
      setConfig(saved);
      setMcpServersText(prettyJson(saved?.mcp_config?.servers || []));
    } finally {
      setSaving(false);
    }
  }

  async function onTestAutoSearch() {
    const query = autoSearchTestQuery.trim();
    if (!query) {
      setAutoSearchTestError('Enter a sample query first.');
      return;
    }
    setAutoSearchTestRunning(true);
    setAutoSearchTestError(null);
    setAutoSearchTestResult(null);
    try {
      const res = await testAutoSearch({
        query,
        force: autoSearchTestForce,
        bypass_cache: autoSearchTestBypass
      });
      setAutoSearchTestResult(res);
    } catch (err) {
      setAutoSearchTestError(err instanceof Error ? err.message : 'Test failed');
    } finally {
      setAutoSearchTestRunning(false);
    }
  }

  async function onValidate() {
    if (!config) {
      return;
    }
    const result = await validateProvider({
      base_url: normalizeBaseUrl(
        (editingProvider?.base_url ?? '') ||
          (activeProviderId
            ? providers.find((p) => p.id === activeProviderId)?.base_url ?? ''
            : '')
      ),
      api_key:
        editingProvider?.api_key ??
        (activeProviderId
          ? providers.find((p) => p.id === activeProviderId)?.api_key ?? ''
          : ''),
      endpoint: '/models',
      timeout_sec: parseBoundedInt(
        editingProvider?.request_timeout_sec ?? 240,
        240,
        5,
        600
      )
    });
    setValidation(result as ValidationState);
  }

  async function onDiscoverMcp() {
    setMcpDiscovery(await discoverMcpTools());
  }

  const sortedProviders = useMemo(
    () =>
      [...providers].sort((a, b) => {
        const aActive = a.id === activeProviderId ? 0 : 1;
        const bActive = b.id === activeProviderId ? 0 : 1;
        if (aActive !== bActive) return aActive - bActive;
        return a.name.localeCompare(b.name);
      }),
    [providers, activeProviderId]
  );

  function startNewProvider() {
    setEditingProvider(emptyEditingProvider());
    setEditingProviderId(null);
  }

  function startEditProvider(p: Provider) {
    setEditingProvider({
      id: p.id,
      name: p.name,
      provider_name: p.provider_name,
      base_url: p.base_url,
      endpoint: p.endpoint,
      api_key: p.api_key,
      request_timeout_sec: p.request_timeout_sec,
      enabled: p.enabled,
      notes: p.notes
    });
    setEditingProviderId(p.id);
  }

  async function onSaveProvider(e: FormEvent) {
    e.preventDefault();
    if (!editingProvider || providerSaving) return;
    setProviderSaving(true);
    setProviderError(null);
    try {
      const payload: ProviderCreateInput = {
        name: editingProvider.name.trim() || 'Untitled provider',
        provider_name: editingProvider.provider_name || 'openai-compatible',
        base_url: normalizeBaseUrl(editingProvider.base_url),
        endpoint: editingProvider.endpoint || '/chat/completions',
        api_key: editingProvider.api_key || '',
        request_timeout_sec: parseBoundedInt(editingProvider.request_timeout_sec, 240, 5, 600),
        enabled: editingProvider.enabled,
        notes: editingProvider.notes || '',
        models: editingProviderId
          ? undefined
          : [
              {
                name: '',
                display_name: '',
                context_window_size: 128000,
                max_output_tokens: 2048,
                is_default: true,
                enabled: true
              }
            ]
      };
      let saved: Provider;
      if (editingProviderId) {
        saved = await updateProvider(editingProviderId, {
          name: payload.name,
          provider_name: payload.provider_name,
          base_url: payload.base_url,
          endpoint: payload.endpoint,
          api_key: payload.api_key,
          request_timeout_sec: payload.request_timeout_sec,
          enabled: payload.enabled,
          notes: payload.notes
        });
      } else {
        saved = await createProvider(payload);
        setEditingProviderId(saved.id);
        // Pre-fill the new model draft so the user can fill in the name
        // field without having to add a separate row.
        if (saved.models.length > 0) {
          setNewModelDraft((prev) => ({
            ...prev,
            [saved.id]: {
              name: '',
              display_name: '',
              context_window_size: saved.models[0].context_window_size,
              max_output_tokens: saved.models[0].max_output_tokens,
              temperature: saved.models[0].temperature,
              top_p: saved.models[0].top_p,
              is_default: saved.models.length === 0,
              enabled: true
            }
          }));
        }
      }
      await refreshProviders();
      setEditingProvider({
        id: saved.id,
        name: saved.name,
        provider_name: saved.provider_name,
        base_url: saved.base_url,
        endpoint: saved.endpoint,
        api_key: saved.api_key,
        request_timeout_sec: saved.request_timeout_sec,
        enabled: saved.enabled,
        notes: saved.notes
      });
    } catch (err) {
      setProviderError(err instanceof Error ? err.message : 'Failed to save provider');
    } finally {
      setProviderSaving(false);
    }
  }

  async function onDeleteProvider(p: Provider) {
    if (!confirm(`Delete provider "${p.name}"? Models under it will also be removed.`)) {
      return;
    }
    setProviderSaving(true);
    setProviderError(null);
    try {
      await deleteProvider(p.id);
      if (editingProviderId === p.id) {
        setEditingProvider(null);
        setEditingProviderId(null);
      }
      await refreshProviders();
    } catch (err) {
      setProviderError(err instanceof Error ? err.message : 'Failed to delete provider');
    } finally {
      setProviderSaving(false);
    }
  }

  async function onValidateProviderDraft(p?: Provider) {
    const draft = editingProvider && (!p || p.id === editingProviderId) ? editingProvider : p;
    if (!draft) return;
    const result = await validateProvider({
      base_url: normalizeBaseUrl(draft.base_url),
      api_key: draft.api_key,
      endpoint: '/models',
      timeout_sec: parseBoundedInt(draft.request_timeout_sec, 240, 5, 600)
    });
    setProviderValidation((prev) => ({
      ...prev,
      [p?.id ?? 'draft']: result as ValidationState
    }));
  }

  async function onAddModel(provider: Provider) {
    const draft = newModelDraft[provider.id] || {
      name: '',
      display_name: '',
      context_window_size: 128000,
      max_output_tokens: 2048,
      is_default: provider.models.length === 0,
      enabled: true
    };
    if (!draft.name.trim()) {
      // Surface the error against *this* provider, not the global
      // providerError slot, so it doesn't get clobbered by the next
      // unrelated save action.
      setModelMessage((prev) => ({
        ...prev,
        [provider.id]: '❌ Model id is required.'
      }));
      return;
    }
    setModelSaving(`add:${provider.id}`);
    setProviderError(null);
    setModelMessage((prev) => ({ ...prev, [provider.id]: '' }));
    try {
      // Explicitly forward every field we care about so we never rely
      // on whatever happens to be on the draft.  ``display_name`` falls
      // back to ``name`` so the row in the UI has a friendly label.
      const created = await addProviderModel(provider.id, {
        name: draft.name.trim(),
        display_name: (draft.display_name || '').trim() || draft.name.trim(),
        context_window_size: parseBoundedInt(draft.context_window_size ?? 128000, 128000, 1, 1_000_000),
        max_output_tokens: parseBoundedInt(draft.max_output_tokens ?? 2048, 2048, 1, 1_000_000),
        temperature: Number(draft.temperature ?? 0.2),
        top_p: Number(draft.top_p ?? 1.0),
        extra_params_json: {},
        is_default: Boolean(draft.is_default),
        enabled: draft.enabled !== false
      });
      setNewModelDraft((prev) => ({ ...prev, [provider.id]: emptyModelDraft() }));
      await refreshProviders();
      setModelMessage((prev) => ({
        ...prev,
        [provider.id]: `✓ Added “${created.display_name || created.name}”.`
      }));
    } catch (err) {
      const detail =
        err instanceof Error
          ? err.message
          : 'Failed to add model';
      setModelMessage((prev) => ({
        ...prev,
        [provider.id]: `❌ ${detail}`
      }));
    } finally {
      setModelSaving(null);
    }
  }

  function startEditModel(m: ModelEntry) {
    setEditingModelDraft((prev) => ({ ...prev, [m.id]: { ...m } }));
  }

  async function onSaveModel(provider: Provider, m: ModelEntry) {
    const draft = editingModelDraft[m.id];
    if (!draft) return;
    setModelSaving(`save:${m.id}`);
    setProviderError(null);
    try {
      await updateProviderModel(provider.id, m.id, {
        name: draft.name,
        display_name: draft.display_name,
        context_window_size: parseBoundedInt(draft.context_window_size, 128000, 1, 1_000_000),
        max_output_tokens: parseBoundedInt(draft.max_output_tokens, 2048, 1, 1_000_000),
        temperature: Number(draft.temperature ?? 0.2),
        top_p: Number(draft.top_p ?? 1.0),
        extra_params_json: draft.extra_params_json || {},
        is_default: draft.is_default,
        enabled: draft.enabled
      });
      await refreshProviders();
      setEditingModelDraft((prev) => {
        const next = { ...prev };
        delete next[m.id];
        return next;
      });
    } catch (err) {
      setProviderError(err instanceof Error ? err.message : 'Failed to save model');
    } finally {
      setModelSaving(null);
    }
  }

  async function onDeleteModel(provider: Provider, m: ModelEntry) {
    if (!confirm(`Delete model "${m.display_name || m.name}"?`)) return;
    setModelSaving(`delete:${m.id}`);
    setProviderError(null);
    try {
      await deleteProviderModel(provider.id, m.id);
      setEditingModelDraft((prev) => {
        const next = { ...prev };
        delete next[m.id];
        return next;
      });
      await refreshProviders();
    } catch (err) {
      setProviderError(err instanceof Error ? err.message : 'Failed to delete model');
    } finally {
      setModelSaving(null);
    }
  }

  async function onActivateModel(provider: Provider, m: ModelEntry) {
    setModelSaving(`activate:${m.id}`);
    setProviderError(null);
    try {
      const result = await activateProviderModel(provider.id, m.id);
      setActiveProviderId(result.active_provider_id);
      setActiveModelId(result.active_model_id);
      await refreshProviders();
    } catch (err) {
      setProviderError(err instanceof Error ? err.message : 'Failed to activate model');
    } finally {
      setModelSaving(null);
    }
  }

  async function onSetActiveProvider(providerId: string) {
    setProviderError(null);
    try {
      const result = await setActiveSelection({ provider_id: providerId });
      setActiveProviderId(result.active_provider_id);
      setActiveModelId(result.active_model_id);
      await refreshProviders();
    } catch (err) {
      setProviderError(err instanceof Error ? err.message : 'Failed to set active provider');
    }
  }

  if (!config) {
    return <main className="simple-page">Loading...</main>;
  }

  return (
    <main className="simple-page">
      <header>
        <h1>Settings</h1>
        <Link href="/">Back</Link>
      </header>

      {providerError ? (
        <p style={{ color: '#b91c1c' }}>{providerError}</p>
      ) : null}

      <section className="settings-providers">
        <h2>Providers &amp; Models</h2>
        <p className="small-muted">
          Manage every OpenAI-compatible provider and the models it hosts. The
          active pair is used by new chats, background runs, and evolution tests
          unless a session pins its own selection.
        </p>

        <div className="provider-list">
          {sortedProviders.length === 0 ? (
            <p className="small-muted">
              No providers yet. Use <strong>Add provider</strong> below to register
              one.
            </p>
          ) : (
            sortedProviders.map((p) => (
              <article key={p.id} className={`provider-card ${p.id === activeProviderId ? 'active' : ''}`}>
                <header>
                  <div>
                    <strong>{p.name}</strong>
                    {p.id === activeProviderId ? (
                      <span className="provider-active-badge">active</span>
                    ) : null}
                    <small className="small-muted">
                      {p.base_url || '(no base URL)'}
                    </small>
                  </div>
                  <div className="provider-actions">
                    <button type="button" onClick={() => startEditProvider(p)}>
                      Edit
                    </button>
                    <button type="button" onClick={() => onValidateProviderDraft(p)}>
                      Test
                    </button>
                    <button
                      type="button"
                      disabled={p.id === activeProviderId}
                      onClick={() => onSetActiveProvider(p.id)}
                    >
                      Make active
                    </button>
                    <button
                      type="button"
                      className="danger"
                      onClick={() => onDeleteProvider(p)}
                    >
                      Delete
                    </button>
                  </div>
                </header>
                {providerValidation[p.id] ? (
                  <p
                    style={{
                      color: providerValidation[p.id]!.ok ? '#0f766e' : '#b91c1c',
                      margin: 0
                    }}
                  >
                    {providerValidation[p.id]!.ok
                      ? `Reachable (${providerValidation[p.id]!.models.length} model(s) discovered)`
                      : `Test failed: ${providerValidation[p.id]!.detail}`}
                  </p>
                ) : null}

                <div className="model-list">
                  {p.models.length === 0 ? (
                    <p className="small-muted">No models yet.</p>
                  ) : (
                    <table>
                      <thead>
                        <tr>
                          <th>Name</th>
                          <th>Model id</th>
                          <th>Context</th>
                          <th>Max out</th>
                          <th>Temp</th>
                          <th>Default</th>
                          <th>Enabled</th>
                          <th />
                        </tr>
                      </thead>
                      <tbody>
                        {p.models.map((m) => {
                          const draft = editingModelDraft[m.id];
                          const isEditing = !!draft;
                          const isActive =
                            p.id === activeProviderId && m.id === activeModelId;
                          return (
                            <tr key={m.id} className={isActive ? 'active' : ''}>
                              {isEditing ? (
                                <>
                                  <td>
                                    <input
                                      value={draft.display_name}
                                      onChange={(e) =>
                                        setEditingModelDraft((prev) => ({
                                          ...prev,
                                          [m.id]: { ...prev[m.id], display_name: e.target.value }
                                        }))
                                      }
                                    />
                                  </td>
                                  <td>
                                    <input
                                      value={draft.name}
                                      onChange={(e) =>
                                        setEditingModelDraft((prev) => ({
                                          ...prev,
                                          [m.id]: { ...prev[m.id], name: e.target.value }
                                        }))
                                      }
                                    />
                                  </td>
                                  <td>
                                    <input
                                      type="number"
                                      min="1"
                                      value={draft.context_window_size}
                                      onChange={(e) =>
                                        setEditingModelDraft((prev) => ({
                                          ...prev,
                                          [m.id]: {
                                            ...prev[m.id],
                                            context_window_size: parseBoundedInt(
                                              e.target.value,
                                              128000,
                                              1,
                                              10_000_000
                                            )
                                          }
                                        }))
                                      }
                                    />
                                  </td>
                                  <td>
                                    <input
                                      type="number"
                                      min="1"
                                      value={draft.max_output_tokens}
                                      onChange={(e) =>
                                        setEditingModelDraft((prev) => ({
                                          ...prev,
                                          [m.id]: {
                                            ...prev[m.id],
                                            max_output_tokens: parseBoundedInt(
                                              e.target.value,
                                              2048,
                                              1,
                                              1_000_000
                                            )
                                          }
                                        }))
                                      }
                                    />
                                  </td>
                                  <td>
                                    <input
                                      type="number"
                                      step="0.05"
                                      value={draft.temperature}
                                      onChange={(e) =>
                                        setEditingModelDraft((prev) => ({
                                          ...prev,
                                          [m.id]: {
                                            ...prev[m.id],
                                            temperature: Number(e.target.value)
                                          }
                                        }))
                                      }
                                    />
                                  </td>
                                  <td>
                                    <input
                                      type="checkbox"
                                      checked={draft.is_default}
                                      onChange={(e) =>
                                        setEditingModelDraft((prev) => ({
                                          ...prev,
                                          [m.id]: {
                                            ...prev[m.id],
                                            is_default: e.target.checked
                                          }
                                        }))
                                      }
                                    />
                                  </td>
                                  <td>
                                    <input
                                      type="checkbox"
                                      checked={draft.enabled}
                                      onChange={(e) =>
                                        setEditingModelDraft((prev) => ({
                                          ...prev,
                                          [m.id]: {
                                            ...prev[m.id],
                                            enabled: e.target.checked
                                          }
                                        }))
                                      }
                                    />
                                  </td>
                                  <td>
                                    <button
                                      type="button"
                                      disabled={modelSaving === `save:${m.id}`}
                                      onClick={() => onSaveModel(p, m)}
                                    >
                                      {modelSaving === `save:${m.id}` ? '...' : 'Save'}
                                    </button>
                                    <button
                                      type="button"
                                      onClick={() =>
                                        setEditingModelDraft((prev) => {
                                          const next = { ...prev };
                                          delete next[m.id];
                                          return next;
                                        })
                                      }
                                    >
                                      Cancel
                                    </button>
                                  </td>
                                </>
                              ) : (
                                <>
                                  <td>{m.display_name || m.name}</td>
                                  <td><code>{m.name}</code></td>
                                  <td>{m.context_window_size}</td>
                                  <td>{m.max_output_tokens}</td>
                                  <td>{m.temperature.toFixed(2)}</td>
                                  <td>{m.is_default ? '★' : ''}</td>
                                  <td>{m.enabled ? 'on' : 'off'}</td>
                                  <td className="model-actions">
                                    {isActive ? (
                                      <span className="provider-active-badge">in use</span>
                                    ) : (
                                      <button
                                        type="button"
                                        disabled={!m.enabled || !p.enabled || modelSaving === `activate:${m.id}`}
                                        onClick={() => onActivateModel(p, m)}
                                      >
                                        Use
                                      </button>
                                    )}
                                    <button type="button" onClick={() => startEditModel(m)}>
                                      Edit
                                    </button>
                                    <button
                                      type="button"
                                      className="danger"
                                      disabled={modelSaving === `delete:${m.id}`}
                                      onClick={() => onDeleteModel(p, m)}
                                    >
                                      Delete
                                    </button>
                                  </td>
                                </>
                              )}
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  )}
                </div>

                <div className="model-add-row">
                  <strong>Add model</strong>
                  <input
                    placeholder="Display name (optional)"
                    value={newModelDraft[p.id]?.display_name ?? ''}
                    onChange={(e) =>
                      setNewModelDraft((prev) => ({
                        ...prev,
                        [p.id]: {
                          ...(prev[p.id] || emptyModelDraft()),
                          display_name: e.target.value
                        }
                      }))
                    }
                  />
                  <input
                    placeholder="model-id (sent to provider)"
                    value={newModelDraft[p.id]?.name ?? ''}
                    onChange={(e) =>
                      setNewModelDraft((prev) => ({
                        ...prev,
                        [p.id]: {
                          ...(prev[p.id] || emptyModelDraft()),
                          name: e.target.value
                        }
                      }))
                    }
                  />
                  <input
                    type="number"
                    placeholder="context window"
                    min="1"
                    value={newModelDraft[p.id]?.context_window_size ?? 128000}
                    onChange={(e) =>
                      setNewModelDraft((prev) => ({
                        ...prev,
                        [p.id]: {
                          ...(prev[p.id] || emptyModelDraft()),
                          context_window_size: parseBoundedInt(
                            e.target.value,
                            128000,
                            1,
                            10_000_000
                          )
                        }
                      }))
                    }
                  />
                  <input
                    type="number"
                    placeholder="max output"
                    min="1"
                    value={newModelDraft[p.id]?.max_output_tokens ?? 2048}
                    onChange={(e) =>
                      setNewModelDraft((prev) => ({
                        ...prev,
                        [p.id]: {
                          ...(prev[p.id] || emptyModelDraft()),
                          max_output_tokens: parseBoundedInt(e.target.value, 2048, 1, 1_000_000)
                        }
                      }))
                    }
                  />
                  <input
                    type="number"
                    step="0.05"
                    placeholder="temp"
                    value={newModelDraft[p.id]?.temperature ?? 0.2}
                    onChange={(e) =>
                      setNewModelDraft((prev) => ({
                        ...prev,
                        [p.id]: {
                          ...(prev[p.id] || emptyModelDraft()),
                          temperature: Number(e.target.value)
                        }
                      }))
                    }
                  />
                  <label className="inline-checkbox">
                    <input
                      type="checkbox"
                      checked={newModelDraft[p.id]?.is_default ?? p.models.length === 0}
                      onChange={(e) =>
                        setNewModelDraft((prev) => ({
                          ...prev,
                          [p.id]: {
                            ...(prev[p.id] || emptyModelDraft()),
                            is_default: e.target.checked
                          }
                        }))
                      }
                    />
                    default
                  </label>
                  <button
                    type="button"
                    disabled={modelSaving === `add:${p.id}`}
                    onClick={() => onAddModel(p)}
                  >
                    {modelSaving === `add:${p.id}` ? '...' : 'Add model'}
                  </button>
                </div>
                {modelMessage[p.id] ? (
                  <p
                    className="model-add-feedback"
                    style={{
                      margin: '6px 0 0',
                      color: modelMessage[p.id].startsWith('✓') ? '#0f766e' : '#b91c1c'
                    }}
                  >
                    {modelMessage[p.id]}
                  </p>
                ) : null}
              </article>
            ))
          )}
        </div>

        <div className="provider-editor">
          <h3>{editingProviderId ? 'Edit provider' : 'Add provider'}</h3>
          {editingProvider ? (
            <form onSubmit={onSaveProvider} className="settings-form">
              <label>
                Name
                <input
                  value={editingProvider.name}
                  onChange={(e) =>
                    setEditingProvider({ ...editingProvider, name: e.target.value })
                  }
                  required
                />
              </label>
              <label>
                Provider kind
                <select
                  value={editingProvider.provider_name || 'openai-compatible'}
                  onChange={(e) =>
                    setEditingProvider({
                      ...editingProvider,
                      provider_name: e.target.value
                    })
                  }
                >
                  {PROVIDER_KIND_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Base URL
                <input
                  value={editingProvider.base_url}
                  onChange={(e) =>
                    setEditingProvider({
                      ...editingProvider,
                      base_url: normalizeBaseUrl(e.target.value)
                    })
                  }
                  placeholder="http://host:port/v1"
                />
              </label>
              <label>
                Chat endpoint
                <input
                  value={editingProvider.endpoint}
                  onChange={(e) =>
                    setEditingProvider({ ...editingProvider, endpoint: e.target.value })
                  }
                />
              </label>
              <label>
                API key
                <input
                  value={editingProvider.api_key}
                  onChange={(e) =>
                    setEditingProvider({ ...editingProvider, api_key: e.target.value })
                  }
                />
              </label>
              <label>
                Request timeout (sec)
                <input
                  type="number"
                  min="5"
                  max="600"
                  value={editingProvider.request_timeout_sec}
                  onChange={(e) =>
                    setEditingProvider({
                      ...editingProvider,
                      request_timeout_sec: parseBoundedInt(e.target.value, 240, 5, 600)
                    })
                  }
                />
              </label>
              <label className="inline-checkbox">
                <input
                  type="checkbox"
                  checked={editingProvider.enabled}
                  onChange={(e) =>
                    setEditingProvider({ ...editingProvider, enabled: e.target.checked })
                  }
                />
                Enabled
              </label>
              <label>
                Notes
                <textarea
                  rows={2}
                  value={editingProvider.notes}
                  onChange={(e) =>
                    setEditingProvider({ ...editingProvider, notes: e.target.value })
                  }
                />
              </label>
              <div className="provider-actions">
                <button type="submit" disabled={providerSaving}>
                  {providerSaving ? 'Saving...' : editingProviderId ? 'Save provider' : 'Create provider'}
                </button>
                <button type="button" onClick={() => onValidateProviderDraft()}>
                  Test connection
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setEditingProvider(null);
                    setEditingProviderId(null);
                  }}
                >
                  Cancel
                </button>
              </div>
              {validation ? (
                <p
                  style={{
                    color: validation.ok ? '#0f766e' : '#b91c1c',
                    margin: 0
                  }}
                >
                  {validation.ok
                    ? `Reachable${validation.models.length ? ` (${validation.models.length} model(s) discovered)` : ''}`
                    : `Test failed: ${validation.detail}`}
                </p>
              ) : null}
            </form>
          ) : (
            <button type="button" onClick={startNewProvider}>
              + Add provider
            </button>
          )}
        </div>
      </section>

      <hr />

      <form onSubmit={onSave} className="settings-form">
        <h2>Global defaults</h2>

        <label>
          Model context window size (override, optional)
          <input
            type="number"
            min="1"
            step="1"
            placeholder="leave empty to use the active model's window"
            value={config.model_context_window_size_override ?? ''}
            onChange={(e) =>
              setConfig((prev: any) => ({
                ...prev,
                model_context_window_size_override: parsePositiveIntOrNull(e.target.value)
              }))
            }
          />
        </label>
        <p className="small-muted">
          Effective context window: min(override, active model window). Empty override = active model window.
        </p>

        <label>
          Initial system prompt
          <textarea
            rows={6}
            value={config.system_prompt || ''}
            onChange={(e) =>
              setConfig((prev: any) => ({
                ...prev,
                system_prompt: e.target.value
              }))
            }
          />
        </label>

        <label>
          Pre-rollover threshold
          <input
            type="number"
            min="0"
            max="1"
            step="0.01"
            value={config.rollover_config.pre_rollover_threshold}
            onChange={(e) =>
              setConfig((prev: any) => ({
                ...prev,
                rollover_config: { ...prev.rollover_config, pre_rollover_threshold: parseThreshold(e.target.value) }
              }))
            }
          />
        </label>

        <label>
          Hard rollover threshold
          <input
            type="number"
            min="0"
            max="1"
            step="0.01"
            value={config.rollover_config.hard_rollover_threshold}
            onChange={(e) =>
              setConfig((prev: any) => ({
                ...prev,
                rollover_config: { ...prev.rollover_config, hard_rollover_threshold: parseThreshold(e.target.value) }
              }))
            }
          />
        </label>

        <hr />

        <label className="inline-checkbox">
          <input
            type="checkbox"
            checked={Boolean(config.mcp_config?.enabled)}
            onChange={(e) =>
              setConfig((prev: any) => ({
                ...prev,
                mcp_config: {
                  ...(prev.mcp_config || {}),
                  enabled: e.target.checked,
                  native_web_search_timeout_sec: prev.mcp_config?.native_web_search_timeout_sec ?? 45,
                  servers: prev.mcp_config?.servers || []
                }
              }))
            }
          />
          Enable MCP tools
        </label>

        <label className="inline-checkbox">
          <input
            type="checkbox"
            checked={Boolean(config.mcp_config?.native_web_search_enabled)}
            onChange={(e) =>
              setConfig((prev: any) => ({
                ...prev,
                mcp_config: {
                  ...(prev.mcp_config || {}),
                  enabled: prev.mcp_config?.enabled ?? true,
                  native_web_search_enabled: e.target.checked,
                  native_web_search_path:
                    prev.mcp_config?.native_web_search_path || 'mcp/web-search-mcp/codex-wrapper.mjs',
                  native_web_search_timeout_sec: prev.mcp_config?.native_web_search_timeout_sec ?? 45,
                  native_web_search_env: prev.mcp_config?.native_web_search_env || {},
                  servers: prev.mcp_config?.servers || []
                }
              }))
            }
          />
          Enable native web-search MCP
        </label>

        <label>
          Native web-search path
          <input
            value={config.mcp_config?.native_web_search_path || 'mcp/web-search-mcp/codex-wrapper.mjs'}
            onChange={(e) =>
              setConfig((prev: any) => ({
                ...prev,
                mcp_config: {
                  ...(prev.mcp_config || {}),
                  enabled: prev.mcp_config?.enabled ?? true,
                  native_web_search_enabled: prev.mcp_config?.native_web_search_enabled ?? false,
                  native_web_search_path: e.target.value,
                  native_web_search_timeout_sec: prev.mcp_config?.native_web_search_timeout_sec ?? 45,
                  native_web_search_env: prev.mcp_config?.native_web_search_env || {},
                  servers: prev.mcp_config?.servers || []
                }
              }))
            }
          />
        </label>

        <label>
          Native web-search timeout (sec)
          <input
            type="number"
            min="3"
            max="600"
            step="1"
            value={config.mcp_config?.native_web_search_timeout_sec ?? 45}
            onChange={(e) =>
              setConfig((prev: any) => ({
                ...prev,
                mcp_config: {
                  ...(prev.mcp_config || {}),
                  enabled: prev.mcp_config?.enabled ?? true,
                  native_web_search_enabled: prev.mcp_config?.native_web_search_enabled ?? false,
                  native_web_search_path:
                    prev.mcp_config?.native_web_search_path || 'mcp/web-search-mcp/codex-wrapper.mjs',
                  native_web_search_timeout_sec: parseBoundedInt(e.target.value, 45, 3, 600),
                  native_web_search_env: prev.mcp_config?.native_web_search_env || {},
                  servers: prev.mcp_config?.servers || []
                }
              }))
            }
          />
        </label>
        <p className="small-muted">
          Applies to native search internals and MCP tool-call response timeout.
        </p>

        <label>
          MCP servers JSON
          <textarea
            rows={12}
            value={mcpServersText}
            onChange={(e) => {
              setMcpServersText(e.target.value);
              setMcpParseError(null);
            }}
          />
        </label>
        <p className="small-muted">
          Example: {'[{"name":"filesystem","command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","/absolute/path"]}]'}
        </p>
        {mcpParseError ? <p style={{ color: '#b91c1c', margin: 0 }}>Invalid MCP JSON: {mcpParseError}</p> : null}

        <button type="button" onClick={onDiscoverMcp}>
          Discover MCP Tools
        </button>
        {mcpDiscovery ? (
          <div className="mcp-discovery">
            <p style={{ color: mcpDiscovery.ok ? '#0f766e' : '#b91c1c', margin: 0 }}>
              {mcpDiscovery.ok ? `Discovered tools: ${mcpDiscovery.tools.length}` : 'MCP discovery has errors'}
            </p>
            {mcpDiscovery.errors.length > 0 ? (
              <pre>{prettyJson(mcpDiscovery.errors)}</pre>
            ) : null}
            {mcpDiscovery.tools.length > 0 ? (
              <pre>{prettyJson(mcpDiscovery.tools)}</pre>
            ) : null}
          </div>
        ) : null}

        <hr />

        <h3>Auto web search — "Google where I don't know"</h3>
        <p className="small-muted">
          The orchestrator runs the auto-search router before the model is
          asked to answer.  When the policy fires, the response is dropped
          into the per-turn prompt as a grounded block — the model then
          quotes it instead of guessing.  The native web-search MCP (above)
          stays the actual search backend.
        </p>

        <label className="inline-checkbox">
          <input
            type="checkbox"
            checked={Boolean(config.mcp_config?.auto_search?.enabled)}
            onChange={(e) =>
              setConfig((prev: any) => ({
                ...prev,
                mcp_config: {
                  ...(prev.mcp_config || {}),
                  auto_search: {
                    ...(prev.mcp_config?.auto_search || {}),
                    enabled: e.target.checked,
                    policy: prev.mcp_config?.auto_search?.policy || 'auto'
                  }
                }
              }))
            }
          />
          Google where you don't know (enable auto-search router)
        </label>

        <label>
          Policy
          <select
            value={config.mcp_config?.auto_search?.policy || 'auto'}
            onChange={(e) =>
              setConfig((prev: any) => ({
                ...prev,
                mcp_config: {
                  ...(prev.mcp_config || {}),
                  auto_search: {
                    ...(prev.mcp_config?.auto_search || {}),
                    policy: e.target.value,
                    enabled: prev.mcp_config?.auto_search?.enabled ?? true
                  }
                }
              }))
            }
          >
            <option value="off">off — never auto-search (model still has the raw MCP tool)</option>
            <option value="auto">auto — heuristic decides per turn (recommended)</option>
            <option value="always">always — search on every user turn</option>
          </select>
        </label>

        <div className="provider-list" style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
          <label>
            Summary max chars
            <input
              type="number"
              min={0}
              max={8000}
              value={config.mcp_config?.auto_search?.summary_max_chars ?? 1200}
              onChange={(e) =>
                setConfig((prev: any) => ({
                  ...prev,
                  mcp_config: {
                    ...(prev.mcp_config || {}),
                    auto_search: {
                      ...(prev.mcp_config?.auto_search || {}),
                      summary_max_chars: parseBoundedInt(e.target.value, 1200, 0, 8000)
                    }
                  }
                }))
              }
            />
          </label>
          <label>
            Max citations
            <input
              type="number"
              min={1}
              max={10}
              value={config.mcp_config?.auto_search?.max_citations ?? 5}
              onChange={(e) =>
                setConfig((prev: any) => ({
                  ...prev,
                  mcp_config: {
                    ...(prev.mcp_config || {}),
                    auto_search: {
                      ...(prev.mcp_config?.auto_search || {}),
                      max_citations: parseBoundedInt(e.target.value, 5, 1, 10)
                    }
                  }
                }))
              }
            />
          </label>
          <label>
            Snippet per source (chars)
            <input
              type="number"
              min={0}
              max={4000}
              value={config.mcp_config?.auto_search?.snippet_per_source_chars ?? 320}
              onChange={(e) =>
                setConfig((prev: any) => ({
                  ...prev,
                  mcp_config: {
                    ...(prev.mcp_config || {}),
                    auto_search: {
                      ...(prev.mcp_config?.auto_search || {}),
                      snippet_per_source_chars: parseBoundedInt(e.target.value, 320, 0, 4000)
                    }
                  }
                }))
              }
            />
          </label>
          <label>
            Cache TTL (sec)
            <input
              type="number"
              min={0}
              max={2592000}
              value={config.mcp_config?.auto_search?.cache_ttl_sec ?? 21600}
              onChange={(e) =>
                setConfig((prev: any) => ({
                  ...prev,
                  mcp_config: {
                    ...(prev.mcp_config || {}),
                    auto_search: {
                      ...(prev.mcp_config?.auto_search || {}),
                      cache_ttl_sec: parseBoundedInt(e.target.value, 21600, 0, 30 * 24 * 3600)
                    }
                  }
                }))
              }
            />
          </label>
          <label>
            Prefer engine
            <input
              placeholder="auto (default) / bing / brave / duckduckgo"
              value={config.mcp_config?.auto_search?.prefer_engine || ''}
              onChange={(e) =>
                setConfig((prev: any) => ({
                  ...prev,
                  mcp_config: {
                    ...(prev.mcp_config || {}),
                    auto_search: {
                      ...(prev.mcp_config?.auto_search || {}),
                      prefer_engine: e.target.value
                    }
                  }
                }))
              }
            />
          </label>
          <label className="inline-checkbox">
            <input
              type="checkbox"
              checked={Boolean(config.mcp_config?.auto_search?.include_full_content ?? false)}
              onChange={(e) =>
                setConfig((prev: any) => ({
                  ...prev,
                  mcp_config: {
                    ...(prev.mcp_config || {}),
                    auto_search: {
                      ...(prev.mcp_config?.auto_search || {}),
                      include_full_content: e.target.checked
                    }
                  }
                }))
              }
            />
            Include full page content (slower, larger prompts)
          </label>
        </div>
        <p className="small-muted">
          A successful search is cached for the configured TTL — the same
          question asked again won't re-hit the search backend.
        </p>

        <details>
          <summary>Test the router with a sample query</summary>
          <div className="provider-list" style={{ marginTop: 8 }}>
            <label>
              Sample user message
              <input
                value={autoSearchTestQuery}
                onChange={(e) => setAutoSearchTestQuery(e.target.value)}
                placeholder="e.g. Who is the CEO of Anthropic?"
              />
            </label>
            <label className="inline-checkbox">
              <input
                type="checkbox"
                checked={autoSearchTestForce}
                onChange={(e) => setAutoSearchTestForce(e.target.checked)}
              />
              Force (ignore policy)
            </label>
            <label className="inline-checkbox">
              <input
                type="checkbox"
                checked={autoSearchTestBypass}
                onChange={(e) => setAutoSearchTestBypass(e.target.checked)}
              />
              Bypass cache
            </label>
            <button
              type="button"
              disabled={autoSearchTestRunning}
              onClick={onTestAutoSearch}
            >
              {autoSearchTestRunning ? 'Searching…' : 'Run router'}
            </button>
            {autoSearchTestError ? (
              <p style={{ color: '#b91c1c', margin: 0 }}>{autoSearchTestError}</p>
            ) : null}
            {autoSearchTestResult ? (
              <AutoSearchTestPreview
                decision={autoSearchTestResult.decision}
                result={autoSearchTestResult.result}
              />
            ) : null}
          </div>
        </details>

        <hr />

        <h3>Context mode — how the prompt is built</h3>
        <p className="small-muted">
          Choose whether the orchestrator should replay the full chat
          history to the model (legacy behaviour) or use the SKILL.state
          runtime (arXiv:2608.26263) that swaps the append-only history
          for a validated (spec, state, observation) bundle whenever a
          registered skill matches the user prompt.
        </p>

        <label>
          Default context mode
          <select
            value={config.context_mode || 'full'}
            onChange={(e) =>
              setConfig((prev: any) => ({
                ...prev,
                context_mode: e.target.value,
              }))
            }
          >
            <option value="full">Full session — replay chat history (default, backward compatible)</option>
            <option value="skill_state">SKILL.state — auto-route to skills, drop the history</option>
          </select>
        </label>

        <p className="small-muted" style={{ marginTop: 8 }}>
          Each session can override this default. Per-turn overrides
          (chat composer) take precedence over both. New chats inherit
          whatever value you save here.
        </p>

        <button disabled={saving}>{saving ? 'Saving...' : 'Save'}</button>
      </form>
    </main>
  );
}

function AutoSearchTestPreview({
  decision,
  result
}: {
  decision: AutoSearchTestResponse['decision'];
  result: AutoSearchTestResponse['result'];
}) {
  if (!decision) {
    return null;
  }
  return (
    <div className="mcp-discovery">
      <p style={{ margin: 0 }}>
        <strong>Decision:</strong> {decision.should_search ? 'search' : 'skip'}{' '}
        <span className="small-muted">(reason: {decision.reason}, policy: {decision.policy})</span>
      </p>
      {!decision.should_search ? null : result ? (
        <>
          <p style={{ margin: 0 }}>
            <strong>Result:</strong>{' '}
            {result.error ? (
              <span style={{ color: '#b91c1c' }}>error: {result.error}</span>
            ) : result.cache_hit ? (
              <span>cache hit ({result.took_ms}ms, engine: {result.engine || '—'})</span>
            ) : (
              <span>
                fresh ({result.took_ms}ms, engine: {result.engine || '—'}, {result.citations.length} citation(s))
              </span>
            )}
          </p>
          {result.citations?.length ? (
            <ul>
              {result.citations.map((c: AutoSearchCitation, idx: number) => (
                <li key={`${idx}-${c.url}`}>
                  <a href={c.url} target="_blank" rel="noreferrer">
                    {c.title || c.url}
                  </a>
                  {c.description ? <div className="small-muted">{c.description}</div> : null}
                </li>
              ))}
            </ul>
          ) : null}
        </>
      ) : null}
    </div>
  );
}

function emptyModelDraft(): ModelEntryInput {
  return {
    name: '',
    display_name: '',
    context_window_size: 128000,
    max_output_tokens: 2048,
    temperature: 0.2,
    top_p: 1.0,
    is_default: false,
    enabled: true
  };
}