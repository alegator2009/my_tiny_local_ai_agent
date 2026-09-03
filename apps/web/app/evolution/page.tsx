'use client';

import Link from 'next/link';

import EvolutionPanel from '@/components/EvolutionPanel';

export default function EvolutionPage() {
  return (
    <main className="simple-page">
      <header className="page-head">
        <div>
          <h1>Project generations</h1>
        </div>
        <nav>
          <Link href="/">Back to chat</Link>
          <Link href="/settings">Settings</Link>
        </nav>
      </header>

      <p className="small-muted">
        Manage evolution runs and switch between lineaged generations of the
        project. Each run spawns a sandboxed copy in <code>evolution/</code>
        that can later be activated, copied to the project root, or deleted.
      </p>

      <EvolutionPanel />
    </main>
  );
}
