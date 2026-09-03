import './globals.css';
import type { ReactNode } from 'react';
import { Inter } from 'next/font/google';

import ThemeToggle from '@/components/ThemeToggle';

const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter',
});

export const metadata = {
  title: 'AI Infinite Session',
  description: 'Local-first infinite agent sessions with rollover and retrieval'
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body>
        <ThemeToggle />
        {children}
      </body>
    </html>
  );
}
