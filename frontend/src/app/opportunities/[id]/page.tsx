'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { getOpportunity } from '@/lib/api';
import { ArrowPathIcon, ArrowLeftIcon, DocumentTextIcon, GlobeAltIcon, ClockIcon, BuildingStorefrontIcon, CheckCircleIcon } from '@heroicons/react/24/outline';
import type { Opportunity } from '@/lib/api';
import React from 'react';

export default function OpportunityDetailPage({ params }: { params: { id: string } }) {
  const [opportunity, setOpportunity] = useState<Opportunity | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchOpportunity = async () => {
      setLoading(true);
      try {
        const data = await getOpportunity(params.id);
        setOpportunity(data);
        setError(null);
      } catch (err) {
        console.error("Error fetching opportunity:", err);
        console.error("Attempted to fetch opportunity with ID:", params.id);
        setError("Failed to load the opportunity. It may not exist or there was a network error.");
      } finally {
        setLoading(false);
      }
    };

    fetchOpportunity();
  }, [params.id]);

  if (loading) {
    return (
      <div className="flex justify-center items-center py-20">
        <ArrowPathIcon className="h-10 w-10 animate-spin text-primary" />
      </div>
    );
  }

  if (error || !opportunity) {
    return (
      <div className="flex flex-col gap-8">
        <Link href="/opportunities" className="flex items-center gap-2 text-primary hover:underline">
          <ArrowLeftIcon className="h-4 w-4" />
          Back to Opportunities
        </Link>
        <div className="alert alert-error">
          <p>{error || "Opportunity not found"}</p>
        </div>
      </div>
    );
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    });
  };

  return (
    <div className="flex flex-col gap-8">
      <Link href="/opportunities" className="flex items-center gap-2 text-primary hover:underline">
        <ArrowLeftIcon className="h-4 w-4" />
        Back to Opportunities
      </Link>
      
      <div>
        <h1 className="text-3xl font-bold">{opportunity.title}</h1>
        <div className="mt-2 flex flex-wrap gap-2">
          {opportunity.categories.map((category, idx) => (
            <span key={idx} className="badge badge-outline">
              {category}
            </span>
          ))}
        </div>
        <p className="text-sm mt-2 text-gray-600 dark:text-gray-400">
          Published: {formatDate(opportunity.published)}
          {opportunity.updated && opportunity.updated !== opportunity.published && 
            ` • Updated: ${formatDate(opportunity.updated)}`}
        </p>
        <div className="mt-2">
          <p className="text-sm">
            Authors: {Array.isArray(opportunity.authors) 
              ? opportunity.authors.join(', ') 
              : typeof opportunity.authors === 'string' 
                ? opportunity.authors 
                : 'Unknown'}
          </p>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <div className="card bg-base-100 shadow-sm">
            <div className="card-body">
              {/* <h2 className="card-title text-xl">Abstract</h2>
              <p className="whitespace-pre-line">{opportunity.summary || 'No abstract available'}</p> */}
              
              {opportunity.layman_explanation && (
                <div className="mt-6">
                  <h3 className="text-lg font-semibold">Simplified Explanation</h3>
                  <p className="mt-2 text-gray-600 dark:text-gray-300">{opportunity.layman_explanation}</p>
                </div>
              )}
              
              <div className="flex flex-wrap gap-4 mt-6">
                {opportunity.arxiv_url && (
                  <a 
                    href={opportunity.arxiv_url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="btn btn-primary btn-sm gap-2"
                  >
                    <GlobeAltIcon className="h-4 w-4" />
                    View on ArXiv
                  </a>
                )}
              </div>
            </div>
          </div>
          
          {(opportunity.steps && opportunity.steps.length > 0) && (
            <div className="card bg-base-100 shadow-sm mt-6">
              <div className="card-body">
                <h2 className="card-title text-xl">Implementation Steps</h2>
                <ol className="mt-4 space-y-4">
                  {opportunity.steps.map((step, idx) => (
                    <li key={idx} className="flex gap-3">
                      <div className="bg-primary rounded-full h-6 w-6 flex items-center justify-center flex-shrink-0 text-white">
                        {idx + 1}
                      </div>
                      <p>{step}</p>
                    </li>
                  ))}
                </ol>
              </div>
            </div>
          )}
          
          {(opportunity.challenges && opportunity.challenges.length > 0) && (
            <div className="card bg-base-100 shadow-sm mt-6">
              <div className="card-body">
                <h2 className="card-title text-xl">Technical Challenges</h2>
                <ul className="mt-4 space-y-2">
                  {opportunity.challenges.map((challenge, idx) => (
                    <li key={idx} className="flex gap-2">
                      <span className="text-red-500">•</span>
                      <p>{challenge}</p>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
          
          {(opportunity.resources && opportunity.resources.length > 0) && (
            <div className="card bg-base-100 shadow-sm mt-6">
              <div className="card-body">
                <h2 className="card-title text-xl">Required Resources</h2>
                <ul className="mt-4 space-y-2">
                  {opportunity.resources.map((resource, idx) => (
                    <li key={idx} className="flex gap-2">
                      <CheckCircleIcon className="h-5 w-5 text-green-500 flex-shrink-0" />
                      <p>{resource}</p>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
        
        <div className="flex flex-col gap-6">
          <div className="card bg-base-100 shadow-sm">
            <div className="card-body">
              <h2 className="card-title text-xl">Opportunity Score</h2>
              
              <div className="stats shadow mt-4">
                <div className="stat">
                  <div className="stat-title">Overall Score</div>
                  <div className="stat-value text-primary">{opportunity.overall_score ? (opportunity.overall_score * 100).toFixed(3) : 'N/A'}</div>
                </div>
              </div>
              
              <div className="mt-2">
                <progress
                  className="progress progress-primary w-full"
                  value={opportunity.overall_score ? opportunity.overall_score * 100 : 0}
                  max="100"
                ></progress>
              </div>
              
              <div className="mt-4 space-y-3">
                <div className="flex justify-between items-center">
                  <span>Technical Feasibility</span>
                  <span className="font-semibold">{opportunity.technical_feasibility_score ? (opportunity.technical_feasibility_score * 10).toFixed(3) : 'N/A'}</span>
                </div>
                <progress
                  className="progress progress-primary w-full"
                  value={opportunity.technical_feasibility_score ? opportunity.technical_feasibility_score * 10 : 0}
                  max="100"
                ></progress>
                
                <div className="flex justify-between items-center mt-2">
                  <span>Market Potential</span>
                  <span className="font-semibold">{opportunity.market_potential_score ? (opportunity.market_potential_score * 100).toFixed(3) : 'N/A'}</span>
                </div>
                <progress
                  className="progress progress-secondary w-full"
                  value={opportunity.market_potential_score ? opportunity.market_potential_score * 100 : 0}
                  max="100"
                ></progress>
                
                <div className="flex justify-between items-center mt-2">
                  <span>Impact</span>
                  <span className="font-semibold">{opportunity.impact_score ? (opportunity.impact_score * 100).toFixed(3) : 'N/A'}</span>
                </div>
                <progress
                  className="progress progress-accent w-full"
                  value={opportunity.impact_score ? opportunity.impact_score * 100 : 0}
                  max="100"
                ></progress>
              </div>
            </div>
          </div>
          
          <div className="card bg-base-100 shadow-sm">
            <div className="card-body">
              <h2 className="card-title text-xl">Market Information</h2>
              
              <div className="mt-4">
                <div className="flex gap-2 items-center text-gray-700 dark:text-gray-300">
                  <ClockIcon className="h-5 w-5 text-primary" />
                  <div className="flex flex-col">
                    <span className="text-sm font-semibold">Time to Market</span>
                    <span>{opportunity.time_to_market || 'Not specified'}</span>
                  </div>
                </div>
                
                {opportunity.target_markets && opportunity.target_markets.length > 0 && (
                  <div className="flex gap-2 items-start mt-4 text-gray-700 dark:text-gray-300">
                    <BuildingStorefrontIcon className="h-5 w-5 text-primary mt-1" />
                    <div className="flex flex-col">
                      <span className="text-sm font-semibold">Target Markets</span>
                      <ul className="list-disc list-inside mt-1 space-y-1">
                        {opportunity.target_markets.map((market, idx) => (
                          <li key={idx}>{market}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 