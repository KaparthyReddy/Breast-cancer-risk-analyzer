// Breast Cancer Predictor - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    console.log('Breast Cancer Predictor initialized');
    
    // Initialize form validation
    initializeFormValidation();
    
    // Initialize smooth scrolling
    initializeSmoothScrolling();
    
    // Initialize result animations
    initializeResultAnimations();
});

// Form Validation
function initializeFormValidation() {
    const form = document.getElementById('predictionForm');
    if (!form) return;
    
    const inputs = form.querySelectorAll('input[type="number"]');
    
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            validateInput(this);
        });
        
        input.addEventListener('blur', function() {
            validateInput(this);
        });
    });
    
    form.addEventListener('submit', function(e) {
        if (!validateForm()) {
            e.preventDefault();
            showAlert('Please fill in all required fields with valid values.', 'danger');
        } else {
            showLoadingState();
        }
    });
}

// Validate individual input
function validateInput(input) {
    const value = parseFloat(input.value);
    const min = parseFloat(input.min) || 0;
    const max = parseFloat(input.max) || Infinity;
    
    // Remove previous validation classes
    input.classList.remove('is-valid', 'is-invalid');
    
    if (input.value === '') {
        input.classList.add('is-invalid');
        return false;
    }
    
    if (isNaN(value) || value < min || value > max) {
        input.classList.add('is-invalid');
        return false;
    }
    
    input.classList.add('is-valid');
    return true;
}

// Validate entire form
function validateForm() {
    const form = document.getElementById('predictionForm');
    if (!form) return true;
    
    const inputs = form.querySelectorAll('input[type="number"]');
    let isValid = true;
    
    inputs.forEach(input => {
        if (!validateInput(input)) {
            isValid = false;
        }
    });
    
    return isValid;
}

// Show loading state
function showLoadingState() {
    const submitBtn = document.querySelector('button[type="submit"]');
    if (!submitBtn) return;
    
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = '<span class="loading"></span> Analyzing...';
    submitBtn.disabled = true;
    
    // Re-enable button after 5 seconds (fallback)
    setTimeout(() => {
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    }, 5000);
}

// Show alert messages
function showAlert(message, type = 'info') {
    // Remove existing alerts
    const existingAlerts = document.querySelectorAll('.alert');
    existingAlerts.forEach(alert => alert.remove());
    
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.textContent = message;
    
    const container = document.querySelector('.container');
    const form = document.querySelector('.prediction-form');
    
    if (container && form) {
        container.insertBefore(alert, form);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            alert.remove();
        }, 5000);
    }
}

// Initialize smooth scrolling
function initializeSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Initialize result animations
function initializeResultAnimations() {
    const resultSection = document.querySelector('.result-section');
    if (!resultSection) return;
    
    // Animate result appearance
    resultSection.style.opacity = '0';
    resultSection.style.transform = 'translateY(20px)';
    
    setTimeout(() => {
        resultSection.style.transition = 'all 0.6s ease';
        resultSection.style.opacity = '1';
        resultSection.style.transform = 'translateY(0)';
    }, 100);
}

// Utility function to format numbers
function formatNumber(num, decimals = 2) {
    return parseFloat(num).toFixed(decimals);
}

// Clear form function
function clearForm() {
    const form = document.getElementById('predictionForm');
    if (!form) return;
    
    const inputs = form.querySelectorAll('input');
    inputs.forEach(input => {
        input.value = '';
        input.classList.remove('is-valid', 'is-invalid');
    });
    
    // Clear any existing alerts
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => alert.remove());
    
    showAlert('Form cleared successfully!', 'info');
}

// Sample data function for testing
function fillSampleData() {
    const sampleData = {
        'mean_radius': '14.127',
        'mean_texture': '19.26',
        'mean_perimeter': '91.19',
        'mean_area': '636.2',
        'mean_smoothness': '0.09393',
        'mean_compactness': '0.11387',
        'mean_concavity': '0.09346',
        'mean_concave_points': '0.05246',
        'mean_symmetry': '0.1812',
        'mean_fractal_dimension': '0.06288',
        'radius_error': '0.4601',
        'texture_error': '1.261',
        'perimeter_error': '3.347',
        'area_error': '35.38',
        'smoothness_error': '0.006389',
        'compactness_error': '0.02619',
        'concavity_error': '0.02376',
        'concave_points_error': '0.01215',
        'symmetry_error': '0.01988',
        'fractal_dimension_error': '0.003371',
        'worst_radius': '16.27',
        'worst_texture': '25.71',
        'worst_perimeter': '108.6',
        'worst_area': '826.4',
        'worst_smoothness': '0.1308',
        'worst_compactness': '0.2695',
        'worst_concavity': '0.2445',
        'worst_concave_points': '0.1268',
        'worst_symmetry': '0.2808',
        'worst_fractal_dimension': '0.08597'
    };
    
    Object.keys(sampleData).forEach(key => {
        const input = document.getElementById(key);
        if (input) {
            input.value = sampleData[key];
            validateInput(input);
        }
    });
    
    showAlert('Sample data loaded successfully!', 'success');
}

// Export functions for global access
window.clearForm = clearForm;
window.fillSampleData = fillSampleData;