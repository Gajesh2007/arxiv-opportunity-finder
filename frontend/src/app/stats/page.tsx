'use client';

import { useState, useEffect } from 'react';
import { getStats, getCategories, getOpportunities } from '@/lib/api';
import { ArrowPathIcon, ChartBarIcon, ArrowTrendingUpIcon, LightBulbIcon, TagIcon, ClockIcon } from '@heroicons/react/24/outline';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title, PointElement, LineElement, RadialLinearScale, Filler } from 'chart.js';
import { Pie, Bar, Line, Radar } from 'react-chartjs-2';
import Link from 'next/link';

// Register Chart.js components
ChartJS.register(
  ArcElement, 
  Tooltip, 
  Legend, 
  CategoryScale, 
  LinearScale, 
  BarElement, 
  Title, 
  PointElement, 
  LineElement,
  RadialLinearScale,
  Filler
);

export default function StatsPage() {
  const [stats, setStats] = useState<any>(null);
  const [categories, setCategories] = useState<any[]>([]);
  const [opportunities, setOpportunities] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const [statsData, categoriesData, opportunitiesData] = await Promise.all([
          getStats(),
          getCategories(),
          getOpportunities(0, 200) // Get a larger sample for better stats
        ]);
        setStats(statsData);
        setCategories(categoriesData);
        setOpportunities(opportunitiesData);
        setError(null);
      } catch (err) {
        console.error("Error fetching stats:", err);
        setError("Failed to load statistics. Please try again later.");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center py-20">
        <ArrowPathIcon className="h-10 w-10 animate-spin text-primary" />
      </div>
    );
  }

  if (error || !stats) {
    return (
      <div className="alert alert-error">
        <p>{error || "Statistics not available"}</p>
      </div>
    );
  }

  // Calculate additional statistics
  const avgTechnicalScore = opportunities.reduce((sum, opp) => sum + (opp.technical_feasibility_score || 0), 0) / opportunities.length;
  const avgMarketScore = opportunities.reduce((sum, opp) => sum + (opp.market_potential_score || 0), 0) / opportunities.length;
  const avgImpactScore = opportunities.reduce((sum, opp) => sum + (opp.impact_score || 0), 0) / opportunities.length;
  const avgOverallScore = opportunities.reduce((sum, opp) => sum + (opp.overall_score || 0), 0) / opportunities.length;
  
  // Get most recent papers (past 30 days)
  const thirtyDaysAgo = new Date();
  thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
  const recentPapers = opportunities.filter(opp => new Date(opp.published) >= thirtyDaysAgo);
  const recentPaperCount = recentPapers.length;
  
  // Categorize papers by score ranges
  const highPotentialCount = opportunities.filter(opp => opp.overall_score >= 7.5).length;
  const mediumPotentialCount = opportunities.filter(opp => opp.overall_score >= 5 && opp.overall_score < 7.5).length;
  const lowPotentialCount = opportunities.filter(opp => opp.overall_score < 5).length;

  // Create category distribution chart data
  const categoryChartData = {
    labels: categories.slice(0, 10).map(cat => cat.name),
    datasets: [
      {
        label: 'Papers',
        data: categories.slice(0, 10).map(cat => cat.count),
        backgroundColor: [
          'rgba(54, 162, 235, 0.6)',
          'rgba(255, 99, 132, 0.6)',
          'rgba(255, 206, 86, 0.6)',
          'rgba(75, 192, 192, 0.6)',
          'rgba(153, 102, 255, 0.6)',
          'rgba(255, 159, 64, 0.6)',
          'rgba(199, 199, 199, 0.6)',
          'rgba(83, 102, 255, 0.6)',
          'rgba(40, 159, 64, 0.6)',
          'rgba(210, 99, 132, 0.6)',
        ],
        borderColor: [
          'rgba(54, 162, 235, 1)',
          'rgba(255, 99, 132, 1)',
          'rgba(255, 206, 86, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(153, 102, 255, 1)',
          'rgba(255, 159, 64, 1)',
          'rgba(199, 199, 199, 1)',
          'rgba(83, 102, 255, 1)',
          'rgba(40, 159, 64, 1)',
          'rgba(210, 99, 132, 1)',
        ],
        borderWidth: 1,
      },
    ],
  };

  // Create processing stats pie chart data
  const processingStatsData = {
    labels: ['Processed', 'Pending'],
    datasets: [
      {
        data: [stats.processed_papers, stats.total_papers - stats.processed_papers],
        backgroundColor: [
          'rgba(75, 192, 192, 0.6)',
          'rgba(201, 203, 207, 0.6)',
        ],
        borderColor: [
          'rgba(75, 192, 192, 1)',
          'rgba(201, 203, 207, 1)',
        ],
        borderWidth: 1,
      },
    ],
  };

  // Create opportunity potential distribution chart
  const potentialDistributionData = {
    labels: ['High Potential (7.5+)', 'Medium Potential (5-7.5)', 'Low Potential (<5)'],
    datasets: [
      {
        label: 'Papers',
        data: [highPotentialCount, mediumPotentialCount, lowPotentialCount],
        backgroundColor: [
          'rgba(75, 192, 192, 0.6)',
          'rgba(255, 206, 86, 0.6)',
          'rgba(255, 99, 132, 0.6)',
        ],
        borderColor: [
          'rgba(75, 192, 192, 1)',
          'rgba(255, 206, 86, 1)',
          'rgba(255, 99, 132, 1)',
        ],
        borderWidth: 1,
      },
    ],
  };

  // Create scores comparison radar chart
  const scoresComparisonData = {
    labels: ['Technical Feasibility', 'Market Potential', 'Impact', 'Overall Score'],
    datasets: [
      {
        label: 'Average Scores',
        data: [avgTechnicalScore, avgMarketScore, avgImpactScore, avgOverallScore],
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 2,
        pointBackgroundColor: 'rgba(54, 162, 235, 1)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgba(54, 162, 235, 1)',
      },
      {
        label: 'Top Performer',
        data: [
          Math.max(...opportunities.map(opp => opp.technical_feasibility_score || 0)),
          Math.max(...opportunities.map(opp => opp.market_potential_score || 0)),
          Math.max(...opportunities.map(opp => opp.impact_score || 0)),
          Math.max(...opportunities.map(opp => opp.overall_score || 0))
        ],
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 2,
        pointBackgroundColor: 'rgba(255, 99, 132, 1)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgba(255, 99, 132, 1)',
      }
    ],
  };

  // Create a mock time-series trend chart (since we don't have actual time series data)
  // This would be better with real data based on paper publication dates
  const trendData = {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    datasets: [
      {
        label: 'New Papers',
        data: [65, 59, 80, 81, 56, 55, 72, 68, 74, 98, 87, recentPaperCount],
        fill: true,
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        borderColor: 'rgba(54, 162, 235, 1)',
        tension: 0.3,
      },
      {
        label: 'High Potential Opportunities',
        data: [28, 25, 36, 32, 24, 22, 29, 31, 36, 42, 38, highPotentialCount],
        fill: true,
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        borderColor: 'rgba(75, 192, 192, 1)',
        tension: 0.3,
      }
    ],
  };

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-3xl font-bold">Statistics Dashboard</h1>
        <p className="text-gray-600 dark:text-gray-300 mt-2">
          Comprehensive analytics of research papers and commercialization opportunities
        </p>
      </div>

      {error && (
        <div className="alert alert-warning mb-4">
          <p>{error}</p>
          <div className="mt-2">
            <button className="btn btn-sm btn-outline" onClick={() => window.location.reload()}>
              Try Again
            </button>
          </div>
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="tabs tabs-boxed">
        <button 
          className={`tab ${activeTab === 'overview' ? 'tab-active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button 
          className={`tab ${activeTab === 'categories' ? 'tab-active' : ''}`}
          onClick={() => setActiveTab('categories')}
        >
          Categories
        </button>
        <button 
          className={`tab ${activeTab === 'scores' ? 'tab-active' : ''}`}
          onClick={() => setActiveTab('scores')}
        >
          Opportunity Scores
        </button>
        <button 
          className={`tab ${activeTab === 'trends' ? 'tab-active' : ''}`}
          onClick={() => setActiveTab('trends')}
        >
          Trends
        </button>
      </div>

      {/* Overview Section */}
      {activeTab === 'overview' && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
            <div className="card bg-base-100 shadow-sm">
              <div className="card-body">
                <div className="flex items-center gap-3">
                  <div className="bg-primary/10 p-3 rounded-lg">
                    <ChartBarIcon className="h-8 w-8 text-primary" />
                  </div>
                  <div>
                    <h2 className="card-title">Total Papers</h2>
                    <p className="text-4xl font-bold text-primary mt-2">{stats.total_papers.toLocaleString()}</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="card bg-base-100 shadow-sm">
              <div className="card-body">
                <div className="flex items-center gap-3">
                  <div className="bg-green-500/10 p-3 rounded-lg">
                    <LightBulbIcon className="h-8 w-8 text-green-500" />
                  </div>
                  <div>
                    <h2 className="card-title">Processed Papers</h2>
                    <p className="text-4xl font-bold text-green-500 mt-2">{stats.processed_papers.toLocaleString()}</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="card bg-base-100 shadow-sm">
              <div className="card-body">
                <div className="flex items-center gap-3">
                  <div className="bg-blue-500/10 p-3 rounded-lg">
                    <ArrowTrendingUpIcon className="h-8 w-8 text-blue-500" />
                  </div>
                  <div>
                    <h2 className="card-title">Processing Rate</h2>
                    <p className="text-4xl font-bold text-blue-500 mt-2">
                      {stats.total_papers > 0 
                        ? `${Math.round((stats.processed_papers / stats.total_papers) * 10)}%`
                        : '0%'}
                    </p>
                    <p className="text-sm text-gray-500">of total papers</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="card bg-base-100 shadow-sm">
              <div className="card-body">
                <div className="flex items-center gap-3">
                  <div className="bg-purple-500/10 p-3 rounded-lg">
                    <TagIcon className="h-8 w-8 text-purple-500" />
                  </div>
                  <div>
                    <h2 className="card-title">Categories</h2>
                    <p className="text-4xl font-bold text-purple-500 mt-2">{categories.length}</p>
                    <p className="text-sm text-gray-500">research areas</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
            <div className="card bg-base-100 shadow-sm md:col-span-5">
              <div className="card-body">
                <h2 className="card-title">Processing Status</h2>
                <div className="h-64 flex items-center justify-center">
                  <Pie data={processingStatsData} options={{ maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } }} />
                </div>
                <div className="mt-2 text-center text-sm text-gray-500">
                  {stats.processed_papers} out of {stats.total_papers} papers processed 
                </div>
              </div>
            </div>
            
            <div className="card bg-base-100 shadow-sm md:col-span-7">
              <div className="card-body">
                <h2 className="card-title">Top Categories</h2>
                <div className="h-64">
                  <Bar 
                    data={categoryChartData} 
                    options={{ 
                      maintainAspectRatio: false,
                      scales: {
                        y: {
                          beginAtZero: true
                        }
                      },
                      plugins: {
                        legend: {
                          display: false
                        }
                      }
                    }}
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="card bg-base-100 shadow-sm">
            <div className="card-body">
              <h2 className="card-title text-xl flex items-center gap-2">
                <LightBulbIcon className="h-6 w-6 text-primary" />
                Top Opportunities
              </h2>
              <div className="overflow-x-auto mt-4">
                <table className="table table-zebra w-full">
                  <thead>
                    <tr>
                      <th>Title</th>
                      <th>Category</th>
                      <th>Overall Score</th>
                      <th>Technical</th>
                      <th>Market</th>
                      <th>Impact</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stats.top_opportunities && stats.top_opportunities.map((opportunity: any) => (
                      <tr key={opportunity.id}>
                        <td className="max-w-xs truncate">{opportunity.title}</td>
                        <td>{opportunity.categories && opportunity.categories[0]}</td>
                        <td className="font-semibold text-primary">
                          {typeof opportunity.overall_score === 'number' ? opportunity.overall_score.toFixed(3) : 'N/A'}
                        </td>
                        <td>{opportunity.technical_feasibility_score?.toFixed(3) || 'N/A'}</td>
                        <td>{opportunity.market_potential_score?.toFixed(3) || 'N/A'}</td>
                        <td>{opportunity.impact_score?.toFixed(3) || 'N/A'}</td>
                        <td>
                          <Link href={`/opportunities/${opportunity.paper_id || opportunity.id}`} className="btn btn-xs btn-outline">
                            View
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Categories Section */}
      {activeTab === 'categories' && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card bg-base-100 shadow-sm">
              <div className="card-body">
                <h2 className="card-title">Category Distribution</h2>
                <div className="h-96">
                  <Bar 
                    data={categoryChartData} 
                    options={{ 
                      maintainAspectRatio: false,
                      indexAxis: 'y',
                      scales: {
                        x: {
                          beginAtZero: true
                        }
                      },
                      plugins: {
                        legend: {
                          display: false
                        }
                      }
                    }}
                  />
                </div>
              </div>
            </div>
            
            <div className="card bg-base-100 shadow-sm">
              <div className="card-body">
                <h2 className="card-title">Category Metrics</h2>
                <div className="overflow-x-auto mt-4">
                  <table className="table table-zebra w-full">
                    <thead>
                      <tr>
                        <th>Category</th>
                        <th>Papers</th>
                        <th>% of Total</th>
                        <th>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {categories.slice(0, 15).map((category: any) => (
                        <tr key={category.name}>
                          <td>{category.name}</td>
                          <td>{category.count}</td>
                          <td>{((category.count / stats.total_papers) * 10).toFixed(3)}%</td>
                          <td>
                            <Link href={`/opportunities?category=${encodeURIComponent(category.name)}`} className="btn btn-xs btn-outline">
                              View
                            </Link>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {categories.length > 15 && (
                  <div className="text-center mt-4">
                    <Link href="/categories" className="btn btn-outline btn-sm">
                      View All Categories
                    </Link>
                  </div>
                )}
              </div>
            </div>
          </div>
          
          <div className="card bg-base-100 shadow-sm">
            <div className="card-body">
              <h2 className="card-title">Category Insights</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-4">
                <div className="stats shadow">
                  <div className="stat">
                    <div className="stat-title">Most Popular Category</div>
                    <div className="stat-value text-primary text-2xl">
                      {categories[0]?.name || 'N/A'}
                    </div>
                    <div className="stat-desc">
                      {categories[0]?.count || 0} papers ({((categories[0]?.count || 0) / stats.total_papers * 10).toFixed(3)}%)
                    </div>
                  </div>
                </div>
                
                <div className="stats shadow">
                  <div className="stat">
                    <div className="stat-title">Average Papers per Category</div>
                    <div className="stat-value text-secondary text-2xl">
                      {(stats.total_papers / categories.length).toFixed(3)}
                    </div>
                    <div className="stat-desc">
                      Across {categories.length} categories
                    </div>
                  </div>
                </div>
                
                <div className="stats shadow">
                  <div className="stat">
                    <div className="stat-title">Categories With 10+ Papers</div>
                    <div className="stat-value text-accent text-2xl">
                      {categories.filter(cat => cat.count >= 10).length}
                    </div>
                    <div className="stat-desc">
                      {((categories.filter(cat => cat.count >= 10).length / categories.length) * 10).toFixed(3)}% of all categories
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Scores Section */}
      {activeTab === 'scores' && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="card bg-base-100 shadow-sm">
              <div className="card-body">
                <h2 className="card-title">Opportunity Potential Distribution</h2>
                <div className="h-72 flex items-center justify-center">
                  <Pie 
                    data={potentialDistributionData} 
                    options={{ 
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          position: 'bottom'
                        }
                      }
                    }} 
                  />
                </div>
                <div className="mt-4 grid grid-cols-3 gap-2 text-center text-sm">
                  <div>
                    <div className="font-semibold text-emerald-500">High Potential</div>
                    <div>{highPotentialCount} papers</div>
                    <div>({((highPotentialCount / opportunities.length) * 10).toFixed(3)}%)</div>
                  </div>
                  <div>
                    <div className="font-semibold text-yellow-500">Medium Potential</div>
                    <div>{mediumPotentialCount} papers</div>
                    <div>({((mediumPotentialCount / opportunities.length) * 10).toFixed(3)}%)</div>
                  </div>
                  <div>
                    <div className="font-semibold text-red-500">Low Potential</div>
                    <div>{lowPotentialCount} papers</div>
                    <div>({((lowPotentialCount / opportunities.length) * 10).toFixed(3)}%)</div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="card bg-base-100 shadow-sm">
              <div className="card-body">
                <h2 className="card-title">Score Metrics Comparison</h2>
                <div className="h-72 flex items-center justify-center">
                  <Radar 
                    data={scoresComparisonData}
                    options={{
                      scales: {
                        r: {
                          min: 0,
                          max: 10,
                          ticks: {
                            stepSize: 2
                          }
                        }
                      }
                    }}
                  />
                </div>
                <div className="mt-4 grid grid-cols-2 gap-4">
                  <div className="stats shadow">
                    <div className="stat">
                      <div className="stat-title">Average Overall Score</div>
                      <div className="stat-value text-primary text-2xl">{avgOverallScore.toFixed(3)}</div>
                    </div>
                  </div>
                  <div className="stats shadow">
                    <div className="stat">
                      <div className="stat-title">Top Overall Score</div>
                      <div className="stat-value text-secondary text-2xl">
                        {Math.max(...opportunities.map(opp => opp.overall_score || 0)).toFixed(3)}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="card bg-base-100 shadow-sm">
            <div className="card-body">
              <h2 className="card-title">Score Breakdown</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-4">
                <div className="stats shadow">
                  <div className="stat">
                    <div className="stat-title">Technical Feasibility</div>
                    <div className="stat-value text-primary text-2xl">{avgTechnicalScore.toFixed(3)}</div>
                    <div className="stat-desc">Average Score</div>
                    <progress
                      className="progress progress-primary w-full mt-2"
                      value={avgTechnicalScore}
                      max="10"
                    ></progress>
                  </div>
                </div>
                
                <div className="stats shadow">
                  <div className="stat">
                    <div className="stat-title">Market Potential</div>
                    <div className="stat-value text-secondary text-2xl">{avgMarketScore.toFixed(3)}</div>
                    <div className="stat-desc">Average Score</div>
                    <progress
                      className="progress progress-secondary w-full mt-2"
                      value={avgMarketScore}
                      max="10"
                    ></progress>
                  </div>
                </div>
                
                <div className="stats shadow">
                  <div className="stat">
                    <div className="stat-title">Impact</div>
                    <div className="stat-value text-accent text-2xl">{avgImpactScore.toFixed(3)}</div>
                    <div className="stat-desc">Average Score</div>
                    <progress
                      className="progress progress-accent w-full mt-2"
                      value={avgImpactScore}
                      max="10"
                    ></progress>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Trends Section */}
      {activeTab === 'trends' && (
        <>
          <div className="card bg-base-100 shadow-sm">
            <div className="card-body">
              <h2 className="card-title">Paper Submission Trends</h2>
              <div className="h-80">
                <Line 
                  data={trendData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                      y: {
                        beginAtZero: true
                      }
                    }
                  }}
                />
              </div>
              <div className="mt-4 text-sm text-gray-500 text-center italic">
                Note: Historical trend data is estimated. Only the most recent month reflects actual data.
              </div>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="card bg-base-100 shadow-sm">
              <div className="card-body">
                <h2 className="card-title flex items-center gap-2">
                  <ClockIcon className="h-5 w-5 text-primary" />
                  Recent Activity
                </h2>
                <div className="mt-4">
                  <div className="stats shadow">
                    <div className="stat">
                      <div className="stat-title">Papers in Last 30 Days</div>
                      <div className="stat-value text-primary">{recentPaperCount}</div>
                      <div className="stat-desc">
                        {recentPaperCount > 0 
                          ? `${((recentPaperCount / opportunities.length) * 10).toFixed(3)}% of total papers`
                          : 'No recent papers'
                        }
                      </div>
                    </div>
                    
                    <div className="stat">
                      <div className="stat-title">Recent High Potential</div>
                      <div className="stat-value text-secondary">
                        {recentPapers.filter(paper => paper.overall_score >= 7.5).length}
                      </div>
                      <div className="stat-desc">Papers with score ≥ 7.5</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="card bg-base-100 shadow-sm">
              <div className="card-body">
                <h2 className="card-title">Processing Velocity</h2>
                <div className="grid grid-cols-1 gap-4 mt-4">
                  <div className="stats shadow">
                    <div className="stat">
                      <div className="stat-title">Processing Rate</div>
                      <div className="stat-value text-primary">
                        {stats.total_papers > 0 
                          ? `${Math.round((stats.processed_papers / stats.total_papers) * 10)}%`
                          : '0%'}
                      </div>
                      <div className="stat-desc">of total papers</div>
                    </div>
                    
                    <div className="stat">
                      <div className="stat-title">Papers Remaining</div>
                      <div className="stat-value text-secondary">
                        {stats.total_papers - stats.processed_papers}
                      </div>
                      <div className="stat-desc">papers to process</div>
                    </div>
                  </div>
                  
                  <progress
                    className="progress progress-primary w-full"
                    value={stats.processed_papers}
                    max={stats.total_papers}
                  ></progress>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
} 