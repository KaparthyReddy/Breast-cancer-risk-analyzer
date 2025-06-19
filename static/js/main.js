document.addEventListener('DOMContentLoaded', function () {
    const formGrid = document.getElementById('formGrid');
    const form = document.getElementById('predictionForm');
    const resultSection = document.getElementById('resultSection');
    const resultTitle = document.getElementById('resultTitle');
    const resultText = document.getElementById('resultText');
    const confidenceFill = document.getElementById('confidenceFill');
    const featureSummary = document.getElementById('featureSummary');

    // Create 30 input fields
    for (let i = 1; i <= 30; i++) {
        const label = document.createElement('label');
        label.innerHTML = `Feature ${i}<input type="number" step="any" name="feature_${i}" required>`;
        formGrid.appendChild(label);
    }

    // Predict
    form.addEventListener('submit', async function (e) {
        e.preventDefault();
        const formData = new FormData(form);
        const jsonData = {};
        formData.forEach((value, key) => {
            jsonData[key] = value;
        });

        const response = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(jsonData)
        });

        const data = await response.json();

        if (data.success) {
            resultSection.style.display = 'block';
            resultTitle.textContent = `Prediction: ${data.prediction}`;
            resultText.textContent = `Confidence: ${data.confidence.toFixed(2)}%`;

            confidenceFill.style.width = `${data.confidence}%`;
            confidenceFill.className = 'confidence-fill ' + (data.prediction === 'Benign' ? 'bg-success' : 'bg-danger');
            confidenceFill.textContent = `${data.confidence.toFixed(1)}%`;

            // Summary
            featureSummary.innerHTML = `<h4>Input Summary</h4><pre>${JSON.stringify(jsonData, null, 2)}</pre>`;
        } else {
            alert('Prediction failed: ' + (data.error || 'Unknown error'));
        }
    });

    // Load Sample
    document.getElementById('loadSample').addEventListener('click', function () {
        const sample = [
            17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419,
            0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373,
            0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622,
            0.6656, 0.7119, 0.2654, 0.4601, 0.1189
        ];

        for (let i = 1; i <= 30; i++) {
            form.elements[`feature_${i}`].value = sample[i - 1];
        }
    });

    // Clear
    document.getElementById('clearBtn').addEventListener('click', function () {
        form.reset();
        resultSection.style.display = 'none';
    });

    // Debug
    document.getElementById('debugBtn').addEventListener('click', function () {
        const debug = {};
        for (let i = 1; i <= 30; i++) {
            debug[`feature_${i}`] = form.elements[`feature_${i}`].value;
        }
        alert(JSON.stringify(debug, null, 2));
    });
});
