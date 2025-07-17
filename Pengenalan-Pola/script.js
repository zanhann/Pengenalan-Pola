// Global variables
let uploadedImage = null;
let currentImageFile = null;

// Waste type mapping
const wasteTypeMapping = {
    "paper": "Kertas",
    "plastic": "Plastik", 
    "metal": "Logam",
    "glass": "Kaca"
};

// Model accuracies (from your Python files)
const modelAccuracies = {
    nn: 74.73,
    nb: 72.15,
    knn: 76.92
};

// DOM elements
const navItems = document.querySelectorAll('.nav-item');
const contentSections = document.querySelectorAll('.content-section');
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const removeImageBtn = document.getElementById('removeImage');
const processBtn = document.getElementById('processBtn');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    initializeFileUpload();
    initializeImageProcessing();
});

// Navigation functionality
function initializeNavigation() {
    navItems.forEach(item => {
        item.addEventListener('click', function() {
            const targetSection = this.getAttribute('data-section');
            switchSection(targetSection);
            
            // Update active nav item
            navItems.forEach(nav => nav.classList.remove('active'));
            this.classList.add('active');
        });
    });
}

function switchSection(sectionId) {
    contentSections.forEach(section => {
        section.classList.remove('active');
    });
    
    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
        targetSection.classList.add('active');
    }
}

// File upload functionality
function initializeFileUpload() {
    // Click to upload
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });
    
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Remove image
    removeImageBtn.addEventListener('click', removeImage);
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file && isValidImageFile(file)) {
        displayImagePreview(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0 && isValidImageFile(files[0])) {
        displayImagePreview(files[0]);
    }
}

function isValidImageFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    const maxSize = 10 * 1024 * 1024; // 10MB
    
    if (!validTypes.includes(file.type)) {
        alert('Format file tidak didukung. Gunakan JPG, JPEG, atau PNG.');
        return false;
    }
    
    if (file.size > maxSize) {
        alert('Ukuran file terlalu besar. Maksimal 10MB.');
        return false;
    }
    
    return true;
}

function displayImagePreview(file) {
    currentImageFile = file;
    
    const reader = new FileReader();
    reader.onload = function(e) {
        uploadedImage = e.target.result;
        previewImg.src = uploadedImage;
        fileName.textContent = `Nama: ${file.name}`;
        fileSize.textContent = `Ukuran: ${formatFileSize(file.size)}`;
        
        uploadArea.style.display = 'none';
        imagePreview.style.display = 'flex';
        processBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function removeImage() {
    uploadedImage = null;
    currentImageFile = null;
    previewImg.src = '';
    fileName.textContent = '';
    fileSize.textContent = '';
    fileInput.value = '';
    
    uploadArea.style.display = 'block';
    imagePreview.style.display = 'none';
    processBtn.disabled = true;
    
    // Hide all prediction results
    hideAllResults();
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Image processing functionality
function initializeImageProcessing() {
    processBtn.addEventListener('click', processImage);
}

function processImage() {
    if (!uploadedImage) {
        alert('Silakan pilih gambar terlebih dahulu.');
        return;
    }
    
    // Show loading state
    processBtn.innerHTML = '<div class="loading"></div> Memproses...';
    processBtn.disabled = true;
    
    // Simulate image processing with different models
    setTimeout(() => {
        simulateImageClassification();
        processBtn.innerHTML = '<i class="fas fa-cogs"></i> Proses Gambar';
        processBtn.disabled = false;
    }, 2000);
}

function simulateImageClassification() {
    // Simulate different predictions from each model
    // In real implementation, this would call your Python backend
    
    const wasteTypes = ['paper', 'plastic', 'metal', 'glass'];
    const wasteTypesIndo = ['Kertas', 'Plastik', 'Logam', 'Kaca'];
    
    // Generate random but realistic predictions
    const predictions = {
        nn: {
            type: wasteTypes[Math.floor(Math.random() * wasteTypes.length)],
            confidence: Math.random() * 30 + 60 // 60-90%
        },
        nb: {
            type: wasteTypes[Math.floor(Math.random() * wasteTypes.length)],
            confidence: Math.random() * 25 + 55 // 55-80%
        },
        knn: {
            type: wasteTypes[Math.floor(Math.random() * wasteTypes.length)],
            confidence: Math.random() * 35 + 50 // 50-85%
        }
    };
    
    // Display results for each model
    displayModelResult('nn', predictions.nn);
    displayModelResult('nb', predictions.nb);
    displayModelResult('knn', predictions.knn);
    
    // Update comparison table
    updateComparisonTable(predictions);
    
    // Determine best prediction
    determineBestPrediction(predictions);
    
    // Show results sections
    showAllResults();
}

function displayModelResult(modelType, prediction) {
    const resultDiv = document.getElementById(`${modelType}Result`);
    const predictionSpan = document.getElementById(`${modelType}Prediction`);
    const confidenceSpan = document.getElementById(`${modelType}Confidence`);
    const confidenceBar = document.getElementById(`${modelType}ConfidenceBar`);
    
    const wasteTypeIndo = wasteTypeMapping[prediction.type] || prediction.type;
    
    predictionSpan.textContent = wasteTypeIndo;
    confidenceSpan.textContent = `${prediction.confidence.toFixed(1)}%`;
    confidenceBar.style.width = `${prediction.confidence}%`;
    
    resultDiv.style.display = 'block';
    
    // Add animation
    resultDiv.style.opacity = '0';
    resultDiv.style.transform = 'translateY(20px)';
    
    setTimeout(() => {
        resultDiv.style.transition = 'all 0.5s ease';
        resultDiv.style.opacity = '1';
        resultDiv.style.transform = 'translateY(0)';
    }, 100);
}

function updateComparisonTable(predictions) {
    document.getElementById('tableNnPrediction').textContent = wasteTypeMapping[predictions.nn.type];
    document.getElementById('tableNnConfidence').textContent = `${predictions.nn.confidence.toFixed(1)}%`;
    
    document.getElementById('tableNbPrediction').textContent = wasteTypeMapping[predictions.nb.type];
    document.getElementById('tableNbConfidence').textContent = `${predictions.nb.confidence.toFixed(1)}%`;
    
    document.getElementById('tableKnnPrediction').textContent = wasteTypeMapping[predictions.knn.type];
    document.getElementById('tableKnnConfidence').textContent = `${predictions.knn.confidence.toFixed(1)}%`;
    
    document.getElementById('comparisonTable').style.display = 'block';
}

function determineBestPrediction(predictions) {
    let bestModel = 'nn';
    let bestScore = predictions.nn.confidence;
    
    // Find model with highest confidence
    Object.keys(predictions).forEach(model => {
        if (predictions[model].confidence > bestScore) {
            bestScore = predictions[model].confidence;
            bestModel = model;
        }
    });
    
    const modelNames = {
        nn: 'Neural Network',
        nb: 'Naive Bayes',
        knn: 'K-Nearest Neighbors'
    };
    
    const bestPrediction = predictions[bestModel];
    
    document.getElementById('bestModel').textContent = modelNames[bestModel];
    document.getElementById('bestPrediction').textContent = `Jenis Sampah: ${wasteTypeMapping[bestPrediction.type]}`;
    document.getElementById('bestConfidence').textContent = `${bestPrediction.confidence.toFixed(1)}%`;
    
    document.getElementById('finalPrediction').style.display = 'block';
}

function showAllResults() {
    // Auto-navigate through sections to show results
    setTimeout(() => switchSection('neural-network'), 500);
    setTimeout(() => switchSection('naive-bayes'), 1500);
    setTimeout(() => switchSection('knn'), 2500);
    setTimeout(() => switchSection('hasil'), 3500);
    
    // Update navigation
    setTimeout(() => {
        navItems.forEach(nav => nav.classList.remove('active'));
        document.querySelector('[data-section="hasil"]').classList.add('active');
    }, 3500);
}

function hideAllResults() {
    document.getElementById('nnResult').style.display = 'none';
    document.getElementById('nbResult').style.display = 'none';
    document.getElementById('knnResult').style.display = 'none';
    document.getElementById('comparisonTable').style.display = 'none';
    document.getElementById('finalPrediction').style.display = 'none';
}

// Utility functions
function showLoading(element) {
    element.innerHTML = '<div class="loading"></div> Memproses...';
    element.disabled = true;
}

function hideLoading(element, originalText) {
    element.innerHTML = originalText;
    element.disabled = false;
}

// Feature extraction simulation (for educational purposes)
function simulateFeatureExtraction(imageData) {
    // This would normally extract actual features from the image
    // For demo purposes, we return random but realistic feature values
    return {
        color_hue_mean: Math.random() * 180,
        color_hue_std: Math.random() * 50,
        color_saturation_mean: Math.random() * 255,
        color_saturation_std: Math.random() * 100,
        color_value_mean: Math.random() * 255,
        color_value_std: Math.random() * 100,
        texture_contrast: Math.random() * 1000,
        texture_dissimilarity: Math.random() * 20,
        texture_homogeneity: Math.random(),
        texture_energy: Math.random(),
        texture_correlation: Math.random(),
        texture_asm: Math.random()
    };
}

// Add smooth scrolling for better UX
function smoothScrollTo(element) {
    element.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
}

// Add keyboard navigation
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        // Close any open modals or reset to home
        switchSection('beranda');
        navItems.forEach(nav => nav.classList.remove('active'));
        document.querySelector('[data-section="beranda"]').classList.add('active');
    }
});

// Add touch support for mobile
let touchStartY = 0;
let touchEndY = 0;

document.addEventListener('touchstart', function(event) {
    touchStartY = event.changedTouches[0].screenY;
});

document.addEventListener('touchend', function(event) {
    touchEndY = event.changedTouches[0].screenY;
    handleSwipe();
});

function handleSwipe() {
    const swipeThreshold = 50;
    const diff = touchStartY - touchEndY;
    
    if (Math.abs(diff) > swipeThreshold) {
        // Add swipe navigation if needed
        console.log(diff > 0 ? 'Swipe up' : 'Swipe down');
    }
}

// Performance optimization
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Add resize handler for responsive design
window.addEventListener('resize', debounce(function() {
    // Handle responsive layout changes
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');
    
    if (window.innerWidth <= 768) {
        // Mobile layout adjustments
        sidebar.style.position = 'relative';
        mainContent.style.marginLeft = '0';
    } else {
        // Desktop layout
        sidebar.style.position = 'fixed';
        mainContent.style.marginLeft = '280px';
    }
}, 250));

// Initialize responsive layout on load
window.addEventListener('load', function() {
    window.dispatchEvent(new Event('resize'));
});

