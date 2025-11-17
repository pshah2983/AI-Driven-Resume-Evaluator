// Main JavaScript for AI-Driven Resume Evaluator

let batchResults = [];

// Tab switching
function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    
    // Hide results when switching tabs
    document.getElementById('results-section').style.display = 'none';
    document.getElementById('batch-results-section').style.display = 'none';
}

// File upload handlers
document.getElementById('resume-file').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        document.getElementById('resume-filename').textContent = file.name;
    }
});

document.getElementById('batch-resume-files').addEventListener('change', function(e) {
    const files = Array.from(e.target.files);
    const container = document.getElementById('batch-filenames');
    container.innerHTML = '';
    
    files.forEach(file => {
        const span = document.createElement('span');
        span.textContent = file.name;
        container.appendChild(span);
    });
});

// Show loading overlay
function showLoading(text = 'Processing...') {
    document.getElementById('loading-text').textContent = text;
    document.getElementById('loading-overlay').style.display = 'flex';
}

// Hide loading overlay
function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}

// Evaluate single resume
async function evaluateResume() {
    const jdText = document.getElementById('jd-text').value.trim();
    const resumeFile = document.getElementById('resume-file').files[0];
    
    if (!jdText) {
        alert('Please enter a job description');
        return;
    }
    
    if (!resumeFile) {
        alert('Please upload a resume file');
        return;
    }
    
    // Create form data
    const formData = new FormData();
    formData.append('jd_text', jdText);
    formData.append('resume_file', resumeFile);
    
    showLoading('Evaluating resume...');
    
    try {
        const response = await fetch('/evaluate', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while evaluating the resume');
    } finally {
        hideLoading();
    }
}

// Display single resume results
function displayResults(data) {
    const scores = data.scores;
    
    // Overall score
    const overallScore = scores.overall_score || 0;
    document.getElementById('overall-score-value').textContent = Math.round(overallScore);
    document.getElementById('confidence-value').textContent = 
        Math.round((scores.confidence || 0) * 100) + '%';
    
    // Score rating
    const rating = getScoreRating(overallScore);
    document.getElementById('score-rating').textContent = rating.emoji + ' ' + rating.text;
    document.getElementById('score-rating').style.color = rating.color;
    
    // Detailed scores
    updateScore('skill', scores.skill_match_score || 0);
    updateScore('exp', scores.experience_match_score || 0);
    updateScore('role', scores.role_responsibility_score || 0);
    updateScore('edu', scores.education_score || 0);
    updateScore('ats', scores.ats_score || 0);
    
    // Skills
    displaySkills('matched-skills', scores.evidence?.matched_skills || [], 'matched');
    displaySkills('missing-skills', scores.missing_skills || [], 'missing');
    
    // Recommendations
    if (scores.summary) {
        document.getElementById('summary').innerHTML = 
            `<h4><i class="fas fa-file-alt"></i> Summary</h4><p>${scores.summary}</p>`;
    }
    
    displayList('gap-analysis', scores.gap_analysis || [], 'ul');
    displayList('action-items', scores.action_items || [], 'ol');
    displayList('interview-questions', scores.interview_questions || [], 'ol');
    
    // Show results section
    document.getElementById('results-section').style.display = 'block';
    document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
}

// Update individual score
function updateScore(type, score) {
    document.getElementById(`${type}-score`).textContent = Math.round(score) + '/100';
    const progress = document.getElementById(`${type}-progress`);
    progress.style.width = score + '%';
}

// Get score rating
function getScoreRating(score) {
    if (score >= 80) {
        return { emoji: '🟢', text: 'Excellent Match', color: '#10b981' };
    } else if (score >= 60) {
        return { emoji: '🟡', text: 'Good Match', color: '#f59e0b' };
    } else {
        return { emoji: '🔴', text: 'Needs Improvement', color: '#ef4444' };
    }
}

// Display skills
function displaySkills(containerId, skills, type) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    if (skills.length === 0) {
        container.innerHTML = '<span style="color: #6b7280;">None</span>';
        return;
    }
    
    skills.slice(0, 20).forEach(skill => {
        const tag = document.createElement('span');
        tag.className = `skill-tag ${type}`;
        tag.textContent = skill;
        container.appendChild(tag);
    });
}

// Display list
function displayList(containerId, items, listType) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    if (items.length === 0) {
        container.innerHTML = '<li style="color: #6b7280;">None</li>';
        return;
    }
    
    items.forEach(item => {
        const li = document.createElement('li');
        li.textContent = item;
        container.appendChild(li);
    });
}

// Evaluate batch resumes
async function evaluateBatch() {
    const jdText = document.getElementById('batch-jd-text').value.trim();
    const resumeFiles = document.getElementById('batch-resume-files').files;
    
    if (!jdText) {
        alert('Please enter a job description');
        return;
    }
    
    if (resumeFiles.length === 0) {
        alert('Please upload at least one resume file');
        return;
    }
    
    // Create form data
    const formData = new FormData();
    formData.append('jd_text', jdText);
    
    Array.from(resumeFiles).forEach(file => {
        formData.append('resume_files', file);
    });
    
    showLoading(`Processing ${resumeFiles.length} resumes...`);
    
    try {
        const response = await fetch('/batch', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }
        
        // Store results
        batchResults = data.results || [];
        
        // Display batch results
        displayBatchResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while processing resumes');
    } finally {
        hideLoading();
    }
}

// Display batch results
function displayBatchResults(data) {
    const results = data.results || [];
    const errors = data.errors || [];
    
    // Summary
    const summaryHtml = `
        <div class="batch-summary-item">
            <div class="batch-summary-label">Total Processed</div>
            <div class="batch-summary-value">${data.total || 0}</div>
        </div>
        <div class="batch-summary-item">
            <div class="batch-summary-label">Successful</div>
            <div class="batch-summary-value" style="color: #10b981;">${data.successful || 0}</div>
        </div>
        <div class="batch-summary-item">
            <div class="batch-summary-label">Errors</div>
            <div class="batch-summary-value" style="color: #ef4444;">${errors.length}</div>
        </div>
    `;
    document.getElementById('batch-summary').innerHTML = summaryHtml;
    
    // Table
    const tbody = document.getElementById('batch-results-tbody');
    tbody.innerHTML = '';
    
    results.forEach((result, index) => {
        const row = document.createElement('tr');
        
        const scoreClass = result.overall_score >= 80 ? 'high' : 
                          result.overall_score >= 60 ? 'medium' : 'low';
        
        row.innerHTML = `
            <td>${index + 1}</td>
            <td>${result.name || 'N/A'}</td>
            <td><span class="score-badge ${scoreClass}">${Math.round(result.overall_score)}</span></td>
            <td>${Math.round(result.skill_score)}</td>
            <td>${Math.round(result.experience_score)}</td>
            <td>${result.years_experience || 0}</td>
            <td>${result.matched_skills || 0}</td>
            <td>${result.missing_skills || 0}</td>
            <td><button class="btn btn-secondary" onclick="viewCandidateDetails(${index})" style="padding: 5px 10px; font-size: 0.9rem;">View</button></td>
        `;
        
        tbody.appendChild(row);
    });
    
    // Show results section
    document.getElementById('batch-results-section').style.display = 'block';
    document.getElementById('batch-results-section').scrollIntoView({ behavior: 'smooth' });
}

// View candidate details
function viewCandidateDetails(index) {
    const candidate = batchResults[index];
    if (!candidate) return;
    
    // Create a modal or detailed view
    const details = `
        Candidate: ${candidate.name}
        Email: ${candidate.email}
        Overall Score: ${candidate.overall_score}
        Skill Score: ${candidate.skill_score}
        Experience Score: ${candidate.experience_score}
        Years Experience: ${candidate.years_experience}
        Matched Skills: ${candidate.matched_skills}
        Missing Skills: ${candidate.missing_skills}
    `;
    
    alert(details);
}

// Export batch results to CSV
function exportBatchCSV() {
    if (batchResults.length === 0) {
        alert('No results to export');
        return;
    }
    
    // Create CSV content
    const headers = ['Rank', 'Name', 'Email', 'Overall Score', 'Skill Score', 'Experience Score', 
                     'Years Experience', 'Matched Skills', 'Missing Skills'];
    
    let csv = headers.join(',') + '\n';
    
    batchResults.forEach((result, index) => {
        const row = [
            index + 1,
            result.name || 'N/A',
            result.email || 'N/A',
            result.overall_score || 0,
            result.skill_score || 0,
            result.experience_score || 0,
            result.years_experience || 0,
            result.matched_skills || 0,
            result.missing_skills || 0
        ];
        csv += row.join(',') + '\n';
    });
    
    // Download CSV
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `batch_evaluation_results_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

