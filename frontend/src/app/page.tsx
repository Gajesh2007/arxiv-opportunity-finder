import Link from 'next/link';
import { BeakerIcon, ChartBarIcon, MagnifyingGlassIcon, LightBulbIcon } from '@heroicons/react/24/outline';

const features = [
  {
    name: 'Discover Opportunities',
    description: 'Find the most promising research papers with high commercial potential.',
    icon: MagnifyingGlassIcon,
  },
  {
    name: 'AI-Powered Analysis',
    description: 'Our AI analyzes papers to identify business opportunities and market potential.',
    icon: BeakerIcon,
  },
  {
    name: 'Market Insights',
    description: 'Get insights on target markets, time to market, and implementation steps.',
    icon: ChartBarIcon,
  },
  {
    name: 'Innovation Tracker',
    description: 'Stay ahead of the curve with the latest research trends and breakthroughs.',
    icon: LightBulbIcon,
  },
];

export default function Home() {
  return (
    <div className="flex flex-col gap-16">
      {/* Hero Section */}
      <div className="relative isolate px-6 pt-14 lg:px-8">
        <div className="mx-auto max-w-2xl py-12 sm:py-20">
          <div className="text-center">
            <h1 className="text-4xl font-bold tracking-tight sm:text-6xl">
              Turn Research into
              <span className="text-primary"> Business Opportunities</span>
            </h1>
            <p className="mt-6 text-lg leading-8 text-gray-600 dark:text-gray-300">
              ArXiv Opportunity Finder helps you discover commercially viable opportunities
              from cutting-edge research papers. Our AI analyzes papers to identify market
              potential and implementation feasibility.
            </p>
            <div className="mt-10 flex items-center justify-center gap-x-6">
              <Link
                href="/opportunities"
                className="btn btn-primary"
              >
                Explore Opportunities
              </Link>
              <Link
                href="/stats"
                className="btn btn-ghost"
              >
                View Stats
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-12 bg-base-200 rounded-xl">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-2xl lg:text-center">
            <h2 className="text-base font-semibold leading-7 text-primary">AI-Powered Discovery</h2>
            <p className="mt-2 text-3xl font-bold tracking-tight sm:text-4xl">
              Everything you need to identify opportunities
            </p>
            <p className="mt-6 text-lg leading-8 text-gray-600 dark:text-gray-300">
              Our platform analyzes arXiv papers using advanced AI to find the most promising
              business opportunities from academic research.
            </p>
          </div>
          <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-4xl">
            <dl className="grid max-w-xl grid-cols-1 gap-x-8 gap-y-10 lg:max-w-none lg:grid-cols-2 lg:gap-y-16">
              {features.map((feature) => (
                <div key={feature.name} className="relative pl-16">
                  <dt className="text-base font-semibold leading-7">
                    <div className="absolute left-0 top-0 flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
                      <feature.icon className="h-6 w-6 text-white" aria-hidden="true" />
                    </div>
                    {feature.name}
                  </dt>
                  <dd className="mt-2 text-base leading-7 text-gray-600 dark:text-gray-300">{feature.description}</dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="py-12">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">
              Ready to discover your next big opportunity?
            </h2>
            <p className="mx-auto mt-6 max-w-xl text-lg leading-8 text-gray-600 dark:text-gray-300">
              Start exploring the latest research papers and find your next business venture.
            </p>
            <div className="mt-10 flex items-center justify-center gap-x-6">
              <Link
                href="/opportunities"
                className="btn btn-primary"
              >
                Get Started
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
