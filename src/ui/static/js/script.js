// Constants for API endpoints
const API_BASE_URL = '';  // Empty means relative to current host
const OPPORTUNITIES_API = `${API_BASE_URL}/opportunities`;
const CATEGORIES_API = `${API_BASE_URL}/categories`;
const STATS_API = `${API_BASE_URL}/stats`;

// DOM elements
const opportunitiesContainer = document.getElementById('opportunities-container');
const minScoreSelect = document.getElementById('min-score');
const sortBySelect = document.getElementById('sort-by');
const sortDirectionSelect = document.getElementById('sort-direction');
const categoryFilterSelect = document.getElementById('category-filter');
const applyFiltersButton = document.getElementById('apply-filters');
const clearFiltersButton = document.getElementById('clear-filters');

// Stats elements
const statsTotal = document.getElementById('stats-total');
const statsProcessed = document.getElementById('stats-processed');
const statsTop = document.getElementById('stats-top');

// Modal elements
const opportunityModal = new bootstrap.Modal(document.getElementById('opportunityModal'));
const modalTitle = document.getElementById('modal-title');
const modalAuthors = document.getElementById('modal-authors');
const modalCategories = document.getElementById('modal-categories');
const modalAbstract = document.getElementById('modal-abstract');
const modalLaymanExplanation = document.getElementById('modal-layman-explanation');
const modalScoreOverall = document.getElementById('modal-score-overall');
const modalScoreInnovation = document.getElementById('modal-score-innovation');
const modalScoreFeasibility = document.getElementById('modal-score-feasibility');
const modalScoreMarket = document.getElementById('modal-score-market');
const modalScoreImpact = document.getElementById('modal-score-impact');
const modalTimeToMarket = document.getElementById('modal-time-to-market');
const modalMarketsList = document.getElementById('modal-markets-list');
const modalPdfLink = document.getElementById('modal-pdf-link');
const modalImplementation = document.getElementById('modal-implementation');
const modalSteps = document.getElementById('modal-steps');
const modalResources = document.getElementById('modal-resources');
const modalChallenges = document.getElementById('modal-challenges');

// Initialize the page
document.addEventListener('DOMContentLoaded', () => {
    // Load initial data
    loadStats();
    loadCategories();
    loadOpportunities();
    
    // Set up event listeners
    applyFiltersButton.addEventListener('click', loadOpportunities);
    clearFiltersButton.addEventListener('click', clearFilters);
});

// Functions to load data
async function loadOpportunities() {
    showLoader();
    
    try {
        // Get filter values
        const minScore = minScoreSelect.value;
        const sortBy = sortBySelect.value;
        const sortDirection = sortDirectionSelect.value;
        const category = categoryFilterSelect.value;
        
        // Build query parameters
        const params = new URLSearchParams();
        params.append('min_score', minScore);
        params.append('sort_by', sortBy);
        params.append('sort_direction', sortDirection);
        if (category) {
            params.append('category', category);
        }
        
        // Fetch opportunities
        const response = await fetch(`${OPPORTUNITIES_API}?${params.toString()}`);
        const data = await response.json();
        
        // Display opportunities
        displayOpportunities(data.opportunities || []);
    } catch (error) {
        console.error('Error loading opportunities:', error);
        opportunitiesContainer.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle"></i> 
                Error loading opportunities. Please try again later.
            </div>
        `;
    }
}

async function loadCategories() {
    try {
        const response = await fetch(CATEGORIES_API);
        const data = await response.json();
        
        // Populate categories dropdown
        const categories = data.categories || [];
        const options = categories.map(category => 
            `<option value="${category}">${category}</option>`
        );
        
        categoryFilterSelect.innerHTML = `
            <option value="" selected>All Categories</option>
            ${options.join('')}
        `;
    } catch (error) {
        console.error('Error loading categories:', error);
    }
}

async function loadStats() {
    try {
        const response = await fetch(STATS_API);
        const data = await response.json();
        
        // Update stats display
        statsTotal.textContent = `${data.total_papers || 0} Total`;
        statsProcessed.textContent = `${data.processed_papers || 0} Processed`;
        statsTop.textContent = `${data.top_opportunities || 0} Top Opportunities`;
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// UI helper functions
function displayOpportunities(opportunities) {
    if (!opportunities || opportunities.length === 0) {
        opportunitiesContainer.innerHTML = `
            <div class="alert alert-info">
                <i class="fas fa-info-circle"></i> 
                No opportunities found matching your criteria. Try adjusting your filters.
            </div>
        `;
        return;
    }
    
    // Create opportunity cards
    let html = '<div class="row">';
    
    opportunities.forEach(opp => {
        // Calculate scores for display (0-100 scale)
        const overallScore = Math.round((opp.combined_score || 0) * 100);
        const innovationScore = Math.round((opp.innovation_score / 10 || 0) * 100);
        const feasibilityScore = Math.round((opp.technical_feasibility_score / 10 || 0) * 100);
        const marketScore = Math.round((opp.market_potential_score / 10 || 0) * 100);
        
        // Determine card border color based on score
        let borderClass = 'border-secondary';
        if (overallScore >= 80) borderClass = 'border-success';
        else if (overallScore >= 70) borderClass = 'border-primary';
        else if (overallScore >= 60) borderClass = 'border-info';
        
        // Format categories
        const categories = Array.isArray(opp.categories) 
            ? opp.categories.join(', ') 
            : opp.categories || '';
        
        html += `
            <div class="col-md-6 col-lg-4 mb-4">
                <div class="card h-100 ${borderClass}" data-paper-id="${opp.paper_id}">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">Score: ${overallScore}%</h5>
                        <span class="badge bg-primary">${categories}</span>
                    </div>
                    <div class="card-body">
                        <h5 class="card-title">${opp.title}</h5>
                        <p class="card-text text-muted small">${truncateText(opp.abstract || '', 150)}</p>
                        
                        <div class="score-grid">
                            <div class="score-item">
                                <span>Innovation</span>
                                <div class="progress">
                                    <div class="progress-bar bg-success" role="progressbar" style="width: ${innovationScore}%" 
                                        aria-valuenow="${innovationScore}" aria-valuemin="0" aria-valuemax="100">
                                        ${Math.round(opp.innovation_score || 0)}/10
                                    </div>
                                </div>
                            </div>
                            <div class="score-item">
                                <span>Feasibility</span>
                                <div class="progress">
                                    <div class="progress-bar bg-info" role="progressbar" style="width: ${feasibilityScore}%" 
                                        aria-valuenow="${feasibilityScore}" aria-valuemin="0" aria-valuemax="100">
                                        ${Math.round(opp.technical_feasibility_score || 0)}/10
                                    </div>
                                </div>
                            </div>
                            <div class="score-item">
                                <span>Market</span>
                                <div class="progress">
                                    <div class="progress-bar bg-warning" role="progressbar" style="width: ${marketScore}%" 
                                        aria-valuenow="${marketScore}" aria-valuemin="0" aria-valuemax="100">
                                        ${Math.round(opp.market_potential_score || 0)}/10
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="card-footer">
                        <button class="btn btn-primary btn-sm view-details" 
                            data-paper-id="${opp.paper_id}">
                            <i class="fas fa-eye"></i> View Details
                        </button>
                        <a href="/papers/${opp.paper_id}/pdf" target="_blank" class="btn btn-outline-secondary btn-sm">
                            <i class="fas fa-file-pdf"></i> PDF
                        </a>
                    </div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    opportunitiesContainer.innerHTML = html;
    
    // Add event listeners to view details buttons
    document.querySelectorAll('.view-details').forEach(button => {
        button.addEventListener('click', () => {
            const paperId = button.getAttribute('data-paper-id');
            showOpportunityDetails(paperId);
        });
    });
}

async function showOpportunityDetails(paperId) {
    try {
        const response = await fetch(`${OPPORTUNITIES_API}/${paperId}`);
        const data = await response.json();
        
        if (!data.opportunity) {
            alert('Opportunity details not found');
            return;
        }
        
        const opp = data.opportunity;
        
        // Fill modal with opportunity details
        modalTitle.textContent = opp.title || '';
        modalAuthors.textContent = opp.authors || '';
        
        // Format categories
        const categories = Array.isArray(opp.categories) 
            ? opp.categories.map(cat => `<span class="badge bg-primary me-1">${cat}</span>`).join(' ') 
            : '';
        modalCategories.innerHTML = categories;
        
        modalAbstract.textContent = opp.abstract || '';
        
        // Set layman's explanation
        modalLaymanExplanation.textContent = opp.layman_explanation || 'No simplified explanation available for this paper.';
        
        // Set scores
        const overallScore = Math.round((opp.combined_score || 0) * 100);
        const innovationScore = Math.round((opp.innovation_score / 10 || 0) * 100);
        const feasibilityScore = Math.round((opp.technical_feasibility_score / 10 || 0) * 100);
        const marketScore = Math.round((opp.market_potential_score / 10 || 0) * 100);
        const impactScore = Math.round((opp.impact_score / 10 || 0) * 100);
        
        modalScoreOverall.style.width = `${overallScore}%`;
        modalScoreOverall.textContent = `${overallScore}%`;
        modalScoreInnovation.style.width = `${innovationScore}%`;
        modalScoreInnovation.textContent = `${innovationScore}%`;
        modalScoreFeasibility.style.width = `${feasibilityScore}%`;
        modalScoreFeasibility.textContent = `${feasibilityScore}%`;
        modalScoreMarket.style.width = `${marketScore}%`;
        modalScoreMarket.textContent = `${marketScore}%`;
        modalScoreImpact.style.width = `${impactScore}%`;
        modalScoreImpact.textContent = `${impactScore}%`;
        
        // Set market info
        modalTimeToMarket.textContent = opp.time_to_market || 'Unknown';
        
        // Handle target markets
        const targetMarkets = opp.target_markets || [];
        if (targetMarkets.length > 0) {
            let marketsHtml = '';
            targetMarkets.forEach(market => {
                marketsHtml += `<li>${market}</li>`;
            });
            modalMarketsList.innerHTML = marketsHtml;
        } else {
            modalMarketsList.innerHTML = '<li>No specific target markets identified</li>';
        }
        
        // Set PDF link
        modalPdfLink.href = `/papers/${opp.paper_id}/pdf`;
        
        // Set implementation details
        modalImplementation.innerHTML = formatLongText(opp.implementation_plan || '');
        
        // Set steps
        const steps = opp.steps || [];
        if (steps.length > 0) {
            let stepsHtml = '';
            steps.forEach(step => {
                stepsHtml += `<li>${step}</li>`;
            });
            modalSteps.innerHTML = stepsHtml;
        } else {
            modalSteps.innerHTML = '<li>No specific steps provided</li>';
        }
        
        // Set resources
        const resources = opp.resources || [];
        if (resources.length > 0) {
            let resourcesHtml = '';
            resources.forEach(resource => {
                resourcesHtml += `<li>${resource}</li>`;
            });
            modalResources.innerHTML = resourcesHtml;
        } else {
            modalResources.innerHTML = '<li>No specific resources listed</li>';
        }
        
        // Set challenges
        const challenges = opp.challenges || [];
        if (challenges.length > 0) {
            let challengesHtml = '';
            challenges.forEach(challenge => {
                challengesHtml += `<li>${challenge}</li>`;
            });
            modalChallenges.innerHTML = challengesHtml;
        } else {
            modalChallenges.innerHTML = '<li>No specific challenges identified</li>';
        }
        
        // Show the modal
        opportunityModal.show();
    } catch (error) {
        console.error('Error loading opportunity details:', error);
        alert('Error loading opportunity details. Please try again.');
    }
}

function clearFilters() {
    minScoreSelect.value = '0';
    sortBySelect.value = 'combined_score';
    sortDirectionSelect.value = 'desc';
    categoryFilterSelect.value = '';
    
    loadOpportunities();
}

function showLoader() {
    opportunitiesContainer.innerHTML = `
        <div class="text-center py-5">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-3">Loading opportunities...</p>
        </div>
    `;
}

function truncateText(text, maxLength) {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

function formatLongText(text) {
    if (!text) return '';
    
    // Convert line breaks to paragraphs
    let formatted = text.split('\n\n')
        .filter(p => p.trim().length > 0)
        .map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`)
        .join('');
    
    return formatted;
} 