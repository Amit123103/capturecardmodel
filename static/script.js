document.addEventListener('DOMContentLoaded', () => {
    const tabUpload = document.getElementById('tab-upload');
    const tabCamera = document.getElementById('tab-camera');
    const panelUpload = document.getElementById('panel-upload');
    const panelCamera = document.getElementById('panel-camera');
    
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    
    const webcam = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const captureBtn = document.getElementById('capture-btn');
    
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const extractBtn = document.getElementById('extract-btn');
    const resetBtn = document.getElementById('reset-btn');
    
    const jsonOutput = document.getElementById('json-output');
    const loading = document.getElementById('loading');
    
    let currentImageBlob = null;
    let stream = null;

    // --- Tab Switching ---
    tabUpload.addEventListener('click', () => {
        tabUpload.classList.add('active');
        tabCamera.classList.remove('active');
        panelUpload.classList.add('active');
        panelCamera.classList.remove('active');
        stopCamera();
    });

    tabCamera.addEventListener('click', () => {
        tabCamera.classList.add('active');
        tabUpload.classList.remove('active');
        panelCamera.classList.add('active');
        panelUpload.classList.remove('active');
        startCamera();
    });

    // --- Camera Logic ---
    async function startCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            webcam.srcObject = stream;
        } catch (err) {
            console.error("Error accessing camera: ", err);
            alert("Could not access the camera. Please ensure permissions are granted.");
        }
    }

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            webcam.srcObject = null;
        }
    }

    captureBtn.addEventListener('click', () => {
        if (!stream) return;
        canvas.width = webcam.videoWidth;
        canvas.height = webcam.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);
        
        canvas.toBlob((blob) => {
            currentImageBlob = blob;
            const imageUrl = URL.createObjectURL(blob);
            showPreview(imageUrl);
            stopCamera();
        }, 'image/jpeg', 0.95);
    });

    // --- Upload Logic ---
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }
        currentImageBlob = file;
        const imageUrl = URL.createObjectURL(file);
        showPreview(imageUrl);
    }

    // --- UI State ---
    function showPreview(url) {
        imagePreview.src = url;
        panelUpload.classList.remove('active');
        panelCamera.classList.remove('active');
        tabUpload.parentElement.style.display = 'none';
        previewContainer.hidden = false;
        jsonOutput.textContent = "Ready to extract.";
    }

    resetBtn.addEventListener('click', () => {
        previewContainer.hidden = true;
        tabUpload.parentElement.style.display = 'flex';
        currentImageBlob = null;
        
        if (tabUpload.classList.contains('active')) {
            panelUpload.classList.add('active');
        } else {
            panelCamera.classList.add('active');
            startCamera();
        }
        
        jsonOutput.textContent = "Ready to extract.";
        loading.hidden = true;
        jsonOutput.hidden = false;
    });

    // --- API Request ---
    extractBtn.addEventListener('click', async () => {
        if (!currentImageBlob) return;

        // UI Loading state
        jsonOutput.hidden = true;
        loading.hidden = false;
        extractBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', currentImageBlob, 'card.jpg');

        // Automatically use localhost when testing locally, and Render when deployed to Vercel
        const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
        const apiUrl = isLocal ? '/extract' : 'https://capturecardmodel.onrender.com/extract';

        try {
            const response = await fetch(apiUrl, {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                // Try to format it if it looks like JSON
                try {
                    const parsed = JSON.parse(result.extracted_text);
                    jsonOutput.textContent = JSON.stringify(parsed, null, 2);
                } catch(e) {
                    jsonOutput.textContent = result.extracted_text;
                }
            } else {
                jsonOutput.textContent = "Error: " + result.error;
            }
        } catch (error) {
            console.error("Error calling API:", error);
            jsonOutput.textContent = "Failed to connect to the server. Is it running?";
        } finally {
            loading.hidden = true;
            jsonOutput.hidden = false;
            extractBtn.disabled = false;
        }
    });
});
