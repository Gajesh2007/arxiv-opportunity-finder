import './globals.css';
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import Navbar from '@/components/layout/Navbar';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'ArXiv Opportunity Finder',
  description: 'Find business opportunities from research papers',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" data-theme="light">
      <body className={inter.className}>
        <div className="min-h-screen flex flex-col">
          <Navbar />
          <main className="flex-grow container mx-auto px-4 py-8">
            {children}
          </main>
          <footer className="bg-base-200 py-6">
            <div className="container mx-auto px-4 text-center text-sm">
              <p>© {new Date().getFullYear()} ArXiv Opportunity Finder</p>
              <p className="mt-2">Find and explore business opportunities from cutting-edge research papers</p>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
