document.getElementById('uploadButton').addEventListener('click', uploadImages);

function uploadImages() {
    const imageInput = document.getElementById('imageInput');
    const files = imageInput.files;

    if (files.length === 0) {
        alert('Please select images to upload.');
        return;
    }

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('images', files[i]);
    }

    fetch('/detect', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        displayResults(data);
    })
    .catch(error => console.error('Error:', error));
}

function displayResults(results) {
    const imageContainer = document.getElementById('imageContainer');
    const resultContainer = document.getElementById('resultContainer');

    imageContainer.innerHTML = '';
    resultContainer.innerHTML = '';

    results.forEach((result, index) => {
        const image = document.createElement('img');
        image.src = URL.createObjectURL(result.image);
        imageContainer.appendChild(image);

        const resultDiv = document.createElement('div');
        resultDiv.className = 'result';

        resultDiv.innerHTML = `
            <h2>Image ${index + 1} Results</h2>
            <ul>
                ${result.detections.map(detection => `
                    <li>
                        Class: ${detection.class}
                        Confidence: ${detection.confidence.toFixed(2)}
                        Bounding Box: (${detection.x1}, ${detection.y1}, ${detection.x2}, ${detection.y2})
                    </li>
                `).join('')}
            </ul>
        `;

        resultContainer.appendChild(resultDiv);
    });
}
const express = require('express');
const multer = require('multer');
const app = express();
const upload = multer({ dest: './uploads/' });

app.post('/detect', upload.array('images'), (req, res) => {
    const images = req.files;
    const results = [];

    // Process each image using your YOLO model
    images.forEach((image, index) => {
        // Load the image and perform detection
        const detections = detectDefects(image.path); // Implement your detection logic here

        results.push({
            image: image.buffer,
            detections: detections.map(detection => ({
                class: detection.class,
                confidence: detection.confidence,
                x1: detection.x1,
                y1: detection.y1,
                x2: detection.x2,
                y2: detection.y2
            }))
        });
    });

    res.json(results);
});

app.listen(3000, () => {
    console.log('Server listening on port 3000');
});
