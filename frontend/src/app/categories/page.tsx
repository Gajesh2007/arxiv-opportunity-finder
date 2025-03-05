'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { getCategories } from '@/lib/api';
import { ArrowPathIcon, TagIcon } from '@heroicons/react/24/outline';

export default function CategoriesPage() {
  const [categories, setCategories] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    const fetchCategories = async () => {
      setLoading(true);
      try {
        const data = await getCategories();
        setCategories(data);
        setError(null);
      } catch (err) {
        console.error("Error fetching categories:", err);
        setError("Failed to load categories. Please try again later.");
      } finally {
        setLoading(false);
      }
    };

    fetchCategories();
  }, []);

  // Filter categories based on search term
  const filteredCategories = categories.filter(category => 
    category && category.name && category.name.toLowerCase().includes((searchTerm || '').toLowerCase())
  );

  // Group categories by first letter for alphabetical display
  const groupedCategories = filteredCategories.reduce((acc, category) => {
    if (!category || !category.name) return acc;
    
    const firstLetter = category.name.charAt(0).toUpperCase();
    if (!acc[firstLetter]) {
      acc[firstLetter] = [];
    }
    acc[firstLetter].push(category);
    return acc;
  }, {} as Record<string, any[]>);

  // Sort the keys alphabetically
  const sortedKeys = Object.keys(groupedCategories).sort();

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-3xl font-bold">Categories</h1>
        <p className="text-gray-600 dark:text-gray-300 mt-2">
          Browse research paper opportunities by category
        </p>
      </div>

      <div className="flex items-center gap-4">
        <div className="form-control flex-1">
          <div className="input-group">
            <input 
              type="text" 
              placeholder="Search categories..." 
              className="input input-bordered flex-1" 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            <button className="btn btn-square">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="flex justify-center items-center py-20">
          <ArrowPathIcon className="h-10 w-10 animate-spin text-primary" />
        </div>
      ) : error ? (
        <div className="alert alert-error">
          <p>{error}</p>
        </div>
      ) : filteredCategories.length === 0 && searchTerm ? (
        <div className="text-center py-20">
          <TagIcon className="h-16 w-16 mx-auto text-gray-400" />
          <h3 className="mt-4 text-lg font-medium">No matching categories found</h3>
          <p className="mt-2 text-gray-600 dark:text-gray-300">
            Try a different search term
          </p>
        </div>
      ) : filteredCategories.length === 0 ? (
        <div className="text-center py-20">
          <Link href="/opportunities" className="btn btn-primary btn-lg">
            Browse All Opportunities
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-8">
          {sortedKeys.map(letter => (
            <div key={letter} className="card bg-base-100 shadow-sm">
              <div className="card-body">
                <h2 className="card-title text-2xl font-bold text-primary">{letter}</h2>
                <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                  {groupedCategories[letter].map(category => (
                    <Link 
                      key={category.name}
                      href={`/opportunities?category=${encodeURIComponent(category.name)}`}
                      className="flex items-center justify-between p-3 rounded-lg border hover:border-primary hover:bg-base-200 transition-colors"
                    >
                      <div className="flex items-center gap-2">
                        <TagIcon className="h-5 w-5 text-primary" />
                        <span>{category.name}</span>
                      </div>
                      <span className="badge badge-primary badge-outline">{category.count}</span>
                    </Link>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
} 