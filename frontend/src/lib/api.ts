import axios from 'axios';

// Create an axios instance with the base URL
const api = axios.create({
  baseURL: `/api`,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
export type Opportunity = {
  id: string;
  paper_id?: string;
  title: string;
  authors: string[];
  summary: string;
  categories: string[];
  published: string;
  updated: string;
  pdf_url: string;
  arxiv_url: string;
  technical_feasibility_score: number;
  market_potential_score: number;
  impact_score: number;
  overall_score: number;
  time_to_market?: string;
  target_markets?: string[];
  steps?: string[];
  resources?: string[];
  challenges?: string[];
  layman_explanation?: string;
  openai_analysis?: any;
};

export type Category = {
  name: string;
  count: number;
};

export type Stats = {
  total_papers: number;
  processed_papers: number;
  top_opportunities: Opportunity[];
};

// API functions
export const getOpportunities = async (
  minScore = 0,
  limit = 100,
  sortBy = 'overall_score',
  sortDirection = 'desc',
  category?: string
): Promise<Opportunity[]> => {
  const params = new URLSearchParams({
    min_score: minScore.toString(),
    limit: limit.toString(),
    sort_by: sortBy,
    sort_direction: sortDirection,
  });
  
  if (category) {
    params.append('category', category);
  }
  
  const response = await api.get(`/opportunities?${params.toString()}`);
  return response.data.opportunities;
};

export const getOpportunity = async (id: string): Promise<Opportunity> => {
  const response = await api.get(`/opportunities/${id}`);
  return response.data.opportunity;
};

export const getCategories = async (): Promise<Category[]> => {
  const response = await api.get('/categories');
  return response.data.categories;
};

export const getStats = async (): Promise<Stats> => {
  const response = await api.get('/stats');
  return response.data;
};

export default api; 