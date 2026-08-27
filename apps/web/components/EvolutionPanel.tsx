'use client';

import { FormEvent, useEffect, useMemo, useState } from 'react';

import {
  activateEvolutionGeneration,
  cancelEvolutionRun,
  copyEvolutionGenerationToRoot,
  deleteEvolutionGeneration,
  listEvolutionEvents,
  listEvolutionGenerations,
  listEvolutionRuns,
  startEvolution,
  type EvolutionEvent,
  type EvolutionGeneration,
  type EvolutionRun
} from '@/lib/api';

function isActive(run: EvolutionRun): boolean {
  return run.status === 'queued' || run.status === 'running';
}

function formatGeneration(run: EvolutionRun): string {
  if (!run.child_generation) {
    return 'agent-???';
  }
  return `agent-${String(run.child_generation).padStart(3, '0')}`;
}

export default function EvolutionPanel() {
  const [prompt, setPrompt] = useState('');
  const [maxGenerations, setMaxGenerations] = useState(1);
  const [mode, setMode] = useState<'conservative' | 'experimental' | 'tests-only'>('conservative');
  const [stopOnFailure, setStopOnFailure] = useState(true);
  const [runs, setRuns] = useState<EvolutionRun[]>([]);
  const [events, setEvents] = useState<EvolutionEvent[]>([]);
  const [generations, setGenerations] = useState<EvolutionGeneration[]>([]);
  const [busy, setBusy] = useState(false);
  const [copyingGeneration, setCopyingGeneration] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);

  const latest = runs[0] || null;
  const activeRun = useMemo(() => runs.find(isActive) || null, [runs]);
  const activeGeneration = useMemo(() => generations.find((generation) => generation.active) || null, [generations]);

  async function reload() {
    const [data, generationData] = await Promise.all([listEvolutionRuns(), listEvolutionGenerations()]);
    setRuns(data);
    setGenerations(generationData);
    const run = data[0];
    if (run) {
      setEvents(await listEvolutionEvents(run.id));
    } else {
      setEvents([]);
    }
  }

  useEffect(() => {
    let cancelled = false;

    async function tick() {
      try {
        const [data, generationData] = await Promise.all([listEvolutionRuns(), listEvolutionGenerations()]);
        if (cancelled) {
          return;
        }
        setRuns(data);
        setGenerations(generationData);
        const run = data[0];
        if (run) {
          const eventData = await listEvolutionEvents(run.id);
          if (!cancelled) {
            setEvents(eventData);
          }
        }
      } catch {
        if (!cancelled) {
          setError('Evolution API unavailable');
        }
      }
    }

    void tick();
    const timer = window.setInterval(() => {
      void tick();
    }, activeRun ? 3000 : 8000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [activeRun?.id]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (busy) {
      return;
    }
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await startEvolution({
        prompt,
        max_generations: maxGenerations,
        mode,
        stop_on_failure: stopOnFailure
      });
      await reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start evolution');
    } finally {
      setBusy(false);
    }
  }

  async function onCancel(runId: string) {
    setError(null);
    setNotice(null);
    try {
      await cancelEvolutionRun(runId);
      await reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel evolution');
    }
  }

  async function onActivate(generation: number) {
    setError(null);
    setNotice(null);
    try {
      await activateEvolutionGeneration(generation);
      await reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to activate generation');
    }
  }

  async function onDelete(generation: EvolutionGeneration) {
    const confirmed = window.confirm(
      generation.active
        ? `Delete active failed generation ${generation.name} from disk?`
        : `Delete ${generation.name} from disk?`
    );
    if (!confirmed) {
      return;
    }
    setError(null);
    setNotice(null);
    try {
      await deleteEvolutionGeneration(generation.generation, generation.active);
      await reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete generation');
    }
  }

  async function onCopyToRoot(generation: EvolutionGeneration) {
    const confirmed = window.confirm(
      `Copy ${generation.name} into root and overwrite root project files? Runtime data, evolution history, node_modules, and virtualenvs will be preserved.`
    );
    if (!confirmed) {
      return;
    }
    setCopyingGeneration(generation.generation);
    setError(null);
    setNotice(null);
    try {
      await copyEvolutionGenerationToRoot(generation.generation);
      setNotice(`${generation.name} copied to root. Restart running dev servers to load root code from disk.`);
      await reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to copy generation to root');
    } finally {
      setCopyingGeneration(null);
    }
  }

  return (
    <section className={`panel evolution-panel ${expanded ? 'expanded' : 'collapsed'}`}>
      <div className="panel-head evolution-head">
        <div>
          <h2>Project generations</h2>
          <small>
            {activeGeneration
              ? `${activeGeneration.name} active · ${activeGeneration.status}`
              : latest
                ? `${formatGeneration(latest)} · ${latest.status}`
                : 'No generations yet'}
          </small>
        </div>
        <div className="evolution-head-actions">
          {latest ? <span className={`evolution-badge ${latest.status}`}>{latest.status}</span> : null}
          <button type="button" onClick={() => setExpanded((value) => !value)}>
            {expanded ? 'Hide' : 'Show'}
          </button>
        </div>
      </div>

      {!expanded ? null : (
        <>
      <form className="evolution-form" onSubmit={onSubmit}>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Task for the next generation"
          rows={2}
        />
        <div className="evolution-controls">
          <label>
            Generations
            <input
              type="number"
              min={1}
              max={20}
              value={maxGenerations}
              onChange={(e) => setMaxGenerations(Math.max(1, Math.min(20, Number(e.target.value) || 1)))}
            />
          </label>
          <label>
            Mode
            <select value={mode} onChange={(e) => setMode(e.target.value as typeof mode)}>
              <option value="conservative">Conservative</option>
              <option value="experimental">Experimental</option>
              <option value="tests-only">Tests only</option>
            </select>
          </label>
          <label className="inline-checkbox evolution-checkbox">
            <input
              type="checkbox"
              checked={stopOnFailure}
              onChange={(e) => setStopOnFailure(e.target.checked)}
            />
            Stop on failure
          </label>
          <button type="submit" disabled={busy || !!activeRun}>
            {busy || activeRun ? 'Working...' : 'Run generation'}
          </button>
        </div>
      </form>

      {error ? <p className="evolution-error">{error}</p> : null}
      {notice ? <p className="evolution-notice">{notice}</p> : null}

      {latest ? (
        <div className="evolution-latest">
          <div>
            <strong>{formatGeneration(latest)}</strong>
            <span>{latest.progress_json?.stage || latest.status}</span>
          </div>
          {latest.child_repo_path ? <code>{latest.child_repo_path}</code> : null}
          {isActive(latest) ? (
            <button type="button" onClick={() => void onCancel(latest.id)}>
              Cancel
            </button>
          ) : null}
        </div>
      ) : null}

      {events.length > 0 ? (
        <ol className="evolution-events">
          {events.slice(-5).map((event) => (
            <li key={event.id}>
              <span>{event.title}</span>
              <small>{event.detail}</small>
            </li>
          ))}
        </ol>
      ) : null}

      {generations.length > 0 ? (
        <div className="evolution-generations">
          <div className="evolution-section-head">
            <strong>Generations</strong>
            <small>{generations.length} total</small>
          </div>
          <ol>
            {generations
              .slice()
              .sort((a, b) => b.generation - a.generation)
              .map((generation) => (
                <li key={generation.generation} className={generation.active ? 'active' : ''}>
                  <div>
                    <strong>{generation.name}</strong>
                    <span className={`evolution-status ${generation.status}`}>{generation.status}</span>
                    {generation.active ? <span className="evolution-active-label">active</span> : null}
                    {generation.has_handoff ? <span className="evolution-active-label">handoff</span> : null}
                  </div>
                  <p>{generation.improvement_summary || generation.prompt || 'No task summary recorded.'}</p>
                  {generation.child_repo_path ? <code>{generation.child_repo_path}</code> : null}
                  <div className="evolution-generation-actions">
                    <button
                      type="button"
                      disabled={generation.active || !generation.child_repo_path}
                      onClick={() => void onActivate(generation.generation)}
                    >
                      Activate
                    </button>
                    <button
                      type="button"
                      disabled={!generation.child_repo_path || !!activeRun || copyingGeneration === generation.generation}
                      onClick={() => void onCopyToRoot(generation)}
                    >
                      {copyingGeneration === generation.generation ? 'Copying...' : 'Copy to root'}
                    </button>
                    <button
                      type="button"
                      disabled={!generation.deletable}
                      onClick={() => void onDelete(generation)}
                    >
                      Delete
                    </button>
                  </div>
                </li>
              ))}
          </ol>
        </div>
      ) : null}
        </>
      )}
    </section>
  );
}
