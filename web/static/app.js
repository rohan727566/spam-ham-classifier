/**
 * Spam Classifier Web UI - JavaScript
 * Author: Rohan Kumar
 */

// Sample messages for testing
const SAMPLE_MESSAGES = {
    ham: [
        "Hey, are we still meeting for lunch tomorrow?",
        "I'll be there in 10 minutes. See you soon!",
        "Thanks for sending the documents. I'll review them tonight.",
        "Meeting rescheduled to 3 PM. Please confirm your attendance."
    ],
    spam: [
        "WINNER! You have won $1000 cash prize. Call now to claim FREE!",
        "Congratulations! You've been selected for a free iPhone. Click here to claim now!",
        "URGENT: Your account has been compromised. Click here to verify immediately!",
        "FREE ENTRY in 2 weekly comp to win FA Cup final tickets. Text FA to 87121."
    ]
};

// DOM Elements
const emailText = document.getElementById('emailText');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const resultsSection = document.getElementById('resultsSection');
const loading = document.getElementById('loading');
const errorMessage = document.getElementById('errorMessage');
const sampleButtons = document.querySelectorAll('.sample-btn');

// Event Listeners
predictBtn.addEventListener('click', predict);
clearBtn.addEventListener('click', clearForm);
emailText.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        predict();
    }
});
sampleButtons.forEach(btn => {
    btn.addEventListener('click', insertSample);
});

/**
 * Get random sample message
 */
function getSampleMessage(type) {
    const messages = SAMPLE_MESSAGES[type];
    return messages[Math.floor(Math.random() * messages.length)];
}

/**
 * Insert sample message into textarea
 */
function insertSample(e) {
    const type = e.target.dataset.type;
    emailText.value = getSampleMessage(type);
    emailText.focus();
}

/**
 * Clear form
 */
function clearForm() {
    emailText.value = '';
    resultsSection.style.display = 'none';
    errorMessage.style.display = 'none';
    emailText.focus();
}

/**
 * Show error message
 */
function showError(message) {
    const errorText = document.getElementById('errorText');
    errorText.textContent = message;
    errorMessage.style.display = 'flex';
    loading.style.display = 'none';
    resultsSection.style.display = 'none';
}

/**
 * Main prediction function
 */
async function predict() {
    const text = emailText.value.trim();

    // Validation
    if (!text) {
        showError('Please enter email text to classify');
        return;
    }

    if (text.length < 5) {
        showError('Please enter at least 5 characters');
        return;
    }

    try {
        // Show loading
        loading.style.display = 'block';
        errorMessage.style.display = 'none';
        resultsSection.style.display = 'none';
        predictBtn.disabled = true;

        // Call API
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }

        const result = await response.json();
        displayResult(result);

    } catch (error) {
        showError('Error: ' + error.message);
    } finally {
        loading.style.display = 'none';
        predictBtn.disabled = false;
    }
}

/**
 * Display prediction results
 */
function displayResult(result) {
    const isSpam = result.is_spam;
    const label = result.label;
    const confidence = result.confidence;
    const probabilities = result.probabilities;
    const cleanedText = result.cleaned_text;

    // Update label and icon
    const labelElement = document.getElementById('resultLabel');
    const labelText = document.getElementById('labelText');
    const labelIcon = document.getElementById('labelIcon');

    labelElement.className = `result-label ${label}`;
    labelText.textContent = label.toUpperCase();
    labelIcon.textContent = isSpam ? '🚫' : '✅';

    // Update confidence
    const confidencePercent = (confidence * 100).toFixed(1);
    const confidenceElement = document.getElementById('resultConfidence');
    confidenceElement.style.background = isSpam ? '#fee2e2' : '#dcfce7';
    confidenceElement.style.color = isSpam ? '#991b1b' : '#15803d';
    confidenceElement.textContent = `${confidencePercent}%`;

    // Update probability bars
    updateProbabilityBar(
        'hamBar',
        'hamValue',
        probabilities.ham
    );
    updateProbabilityBar(
        'spamBar',
        'spamValue',
        probabilities.spam
    );

    // Update cleaned text
    document.getElementById('cleanedText').textContent = cleanedText || '(No words after preprocessing)';

    // Show results
    resultsSection.style.display = 'block';
    loading.style.display = 'none';
    errorMessage.style.display = 'none';
}

/**
 * Update probability bar
 */
function updateProbabilityBar(barId, valueId, probability) {
    const bar = document.getElementById(barId);
    const value = document.getElementById(valueId);

    const percentage = (probability * 100).toFixed(1);

    // Animate bar width
    setTimeout(() => {
        bar.style.width = percentage + '%';
    }, 100);

    value.textContent = percentage + '%';
}

/**
 * Initialize on page load
 */
document.addEventListener('DOMContentLoaded', () => {
    // Check API health
    checkHealth();

    // Focus on textarea
    emailText.focus();
});

/**
 * Check API health
 */
async function checkHealth() {
    try {
        const response = await fetch('/health');
        if (!response.ok) {
            console.warn('API health check failed');
        }
    } catch (error) {
        console.error('Cannot connect to API:', error);
    }
}
