'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { getOpportunities, getCategories } from '@/lib/api';
import { ArrowPathIcon, AdjustmentsHorizontalIcon, FunnelIcon } from '@heroicons/react/24/outline';
import type { Opportunity, Category } from '@/lib/api';

const sortOptions = [
  { label: 'Overall Score', value: 'overall_score' },
  { label: 'Technical Feasibility', value: 'technical_feasibility_score' },
  { label: 'Market Potential', value: 'market_potential_score' },
  { label: 'Impact', value: 'impact_score' },
  { label: 'Publication Date', value: 'published' },
];

export default function OpportunitiesPage() {
  const [opportunities, setOpportunities] = useState<Opportunity[]>([]);
  const [categories, setCategories] = useState<Category[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Filter state
  const [selectedCategory, setSelectedCategory] = useState<string>('');
  const [minScore, setMinScore] = useState<number>(0);
  const [sortBy, setSortBy] = useState<string>('overall_score');
  const [sortDirection, setSortDirection] = useState<string>('desc');
  const [showFilters, setShowFilters] = useState(false);

  const loadData = async () => {
    setLoading(true);
    try {
      // Convert minScore from 0-100 scale to 0-1 scale for the API
      const minScoreForApi = minScore / 100;
      
      const [opportunitiesData, categoriesData] = await Promise.all([
        getOpportunities(minScoreForApi, 100, sortBy, sortDirection, selectedCategory),
        getCategories()
      ]);
      setOpportunities(opportunitiesData);
      setCategories(categoriesData);
      setError(null);
    } catch (err) {
      console.error("Error loading data:", err);
      setError("Failed to load opportunities. Please try again later.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [minScore, sortBy, sortDirection, selectedCategory]);

  const handleCategoryChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedCategory(e.target.value);
  };

  const handleMinScoreChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setMinScore(parseFloat(e.target.value));
  };

  const handleSortByChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSortBy(e.target.value);
  };

  const handleSortDirectionChange = () => {
    setSortDirection(prevDirection => prevDirection === 'asc' ? 'desc' : 'asc');
  };

  const toggleFilters = () => {
    setShowFilters(!showFilters);
  };

  return (
    <div className="flex flex-col gap-8">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold">Research Opportunities</h1>
          <p className="text-gray-600 dark:text-gray-300 mt-2">
            Discover commercial opportunities from cutting-edge research papers
          </p>
        </div>
        
        <button 
          className="btn btn-outline gap-2"
          onClick={toggleFilters}
        >
          <AdjustmentsHorizontalIcon className="h-5 w-5" />
          Filters
        </button>
      </div>

      {showFilters && (
        <div className="bg-base-200 p-4 rounded-xl">
          <div className="text-lg font-semibold mb-3">Filter & Sort</div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div key="category-filter" className="form-control">
              <label className="label">
                <span className="label-text">Category</span>
              </label>
              <select 
                className="select select-bordered w-full" 
                value={selectedCategory}
                onChange={handleCategoryChange}
              >
                <option value="">All Categories</option>
                {categories.map(category => (
                  <option key={category.name} value={category.name}>
                    {category.name} ({category.count})
                  </option>
                ))}
              </select>
            </div>
            
            <div key="score-filter" className="form-control">
              <label className="label">
                <span className="label-text">Minimum Score: {minScore}</span>
              </label>
              <input 
                type="range" 
                min="0" 
                max="100" 
                step="5"
                value={minScore} 
                onChange={handleMinScoreChange}
                className="range range-primary" 
              />
            </div>
            
            <div key="sort-by" className="form-control">
              <label className="label">
                <span className="label-text">Sort By</span>
              </label>
              <select 
                className="select select-bordered w-full" 
                value={sortBy}
                onChange={handleSortByChange}
              >
                {sortOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            
            <div key="sort-direction" className="form-control">
              <label className="label">
                <span className="label-text">Sort Direction</span>
              </label>
              <button 
                className="btn btn-outline w-full"
                onClick={handleSortDirectionChange}
              >
                {sortDirection === 'desc' ? 'Highest First' : 'Lowest First'}
              </button>
            </div>
          </div>
        </div>
      )}
      
      {loading ? (
        <div className="flex justify-center items-center py-20">
          <ArrowPathIcon className="h-10 w-10 animate-spin text-primary" />
        </div>
      ) : error ? (
        <div className="alert alert-error">
          <p>{error}</p>
        </div>
      ) : opportunities.length === 0 ? (
        <div className="text-center py-20">
          <FunnelIcon className="h-16 w-16 mx-auto text-gray-400" />
          <h3 className="mt-4 text-lg font-medium">No opportunities found</h3>
          <p className="mt-2 text-gray-600 dark:text-gray-300">
            Try adjusting your filters to see more results.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {opportunities.map((opportunity, index) => (
            <Link 
              href={`/opportunities/${opportunity.paper_id || opportunity.id}`} 
              key={`${opportunity.paper_id || opportunity.id}-${index}`}
              className="card bg-base-100 hover:shadow-lg transition-shadow duration-300 h-full flex flex-col"
            >
              <div className="card-body flex flex-col h-full">
                <h2 className="card-title text-lg font-bold line-clamp-2">{opportunity.title}</h2>
                
                <div className="flex flex-wrap gap-1 mt-2">
                  {opportunity.categories && opportunity.categories.slice(0, 3).map((category, idx) => (
                    <span 
                      key={`${opportunity.id}-category-${idx}`} 
                      className="badge badge-outline text-xs"
                    >
                      {category}
                    </span>
                  ))}
                  {opportunity.categories && opportunity.categories.length > 3 && (
                    <span className="badge badge-outline text-xs">+{opportunity.categories.length - 3} more</span>
                  )}
                </div>
                
                {/* <p className="text-sm mt-3 line-clamp-3 text-gray-600 dark:text-gray-300">
                  {opportunity.summary || "No abstract available"}
                </p> */}
                
                <div className="mt-auto pt-4 grid grid-cols-3 gap-2">
                  <div key={`${opportunity.id}-tech`} className="flex flex-col">
                    <span className="text-xs text-gray-500">Technical</span>
                    <div className="flex items-center gap-1">
                      <progress 
                        className="progress progress-primary w-full" 
                        value={opportunity.technical_feasibility_score ? opportunity.technical_feasibility_score * 10 : 0} 
                        max="100"
                      ></progress>
                      <span className="text-xs font-semibold">{opportunity.technical_feasibility_score ? (opportunity.technical_feasibility_score * 10).toFixed(3) : 'N/A'}</span>
                    </div>
                  </div>
                  
                  <div key={`${opportunity.id}-market`} className="flex flex-col">
                    <span className="text-xs text-gray-500">Market</span>
                    <div className="flex items-center gap-1">
                      <progress 
                        className="progress progress-secondary w-full" 
                        value={opportunity.market_potential_score ? opportunity.market_potential_score * 10 : 0} 
                        max="100"
                      ></progress>
                      <span className="text-xs font-semibold">{opportunity.market_potential_score ? (opportunity.market_potential_score * 10).toFixed(3) : 'N/A'}</span>
                    </div>
                  </div>
                  
                  <div key={`${opportunity.id}-impact`} className="flex flex-col">
                    <span className="text-xs text-gray-500">Impact</span>
                    <div className="flex items-center gap-1">
                      <progress 
                        className="progress progress-accent w-full" 
                        value={opportunity.impact_score ? opportunity.impact_score * 10 : 0} 
                        max="100"
                      ></progress>
                      <span className="text-xs font-semibold">{opportunity.impact_score ? (opportunity.impact_score * 10).toFixed(3) : 'N/A'}</span>
                    </div>
                  </div>
                </div>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
} 