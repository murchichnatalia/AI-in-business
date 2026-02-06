// app.js (ES module version using transformers.js for local sentiment classification)

import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6/dist/transformers.min.js";

// ========== ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ И ФУНКЦИЯ ЛОГГИРОВАНИЯ ==========
// Global variables
let reviews = [];
let apiToken = "";
let sentimentPipeline = null;

// URL Google Apps Script (🚨 УБЕДИТЕСЬ ЧТО ОН ПРАВИЛЬНЫЙ!)
const GOOGLE_SCRIPT_URL = 'https://script.google.com/macros/s/AKfycbw9nuasR1fCHWmp2zc0okzeIMLrYbojDWyiYRAAH6UnkelkO8Dp4EItNkhxlK5JvsM/exec';

// Функция логирования - теперь она глобальная
async function logToGoogleSheet(review, sentimentLabel, confidenceScore, meta = {}) {
    if (!GOOGLE_SCRIPT_URL) {
        console.warn('⚠️ Google Script URL не настроен.');
        return;
    }

    try {
        const logData = {
            ts_iso: new Date().toISOString(),
            review: review.substring(0, 500),
            sentiment: `${sentimentLabel} (${(confidenceScore * 100).toFixed(1)}%)`,
            meta: JSON.stringify({
                userAgent: navigator.userAgent,
                platform: navigator.platform,
                language: navigator.language,
                screenWidth: window.innerWidth,
                screenHeight: window.innerHeight,
                ...meta
            })
        };

        console.log('📤 Отправляю данные:', { review: review.substring(0, 100), sentimentLabel });

        const response = await fetch(GOOGLE_SCRIPT_URL, {
            method: 'POST',
            mode: 'cors',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(logData)
        });

        const result = await response.text();
        console.log('✅ Ответ от Google:', result);

    } catch (error) {
        console.error('❌ Ошибка при отправке:', error);
    }
}
// ========== КОНЕЦ СЕКЦИИ ЛОГГИРОВАНИЯ ==========

// DOM elements
const analyzeBtn = document.getElementById("analyze-btn");
const reviewText = document.getElementById("review-text");
const sentimentResult = document.getElementById("sentiment-result");
const loadingElement = document.querySelector(".loading");
const errorElement = document.getElementById("error-message");
const apiTokenInput = document.getElementById("api-token");
const statusElement = document.getElementById("status");

// Initialize the app
document.addEventListener("DOMContentLoaded", function () {
  // Load the TSV file (Papa Parse)
  loadReviews();

  // Set up event listeners
  analyzeBtn.addEventListener("click", analyzeRandomReview);
  apiTokenInput.addEventListener("change", saveApiToken);

  // Load saved API token if exists
  const savedToken = localStorage.getItem("hfApiToken");
  if (savedToken) {
    apiTokenInput.value = savedToken;
    apiToken = savedToken;
  }

  // Initialize transformers.js sentiment model
  initSentimentModel();
});

// ... остальной код БЕЗ ИЗМЕНЕНИЙ (initSentimentModel, loadReviews, saveApiToken, 
// analyzeRandomReview, analyzeSentiment) остается точно таким же ...

// Display sentiment result
function displaySentiment(result) {
  // Default to neutral if we can't parse the result
  let sentiment = "neutral";
  let score = 0.5;
  let label = "NEUTRAL";

  // Expected format: [[{label: 'POSITIVE', score: 0.99}]]
  if (
    Array.isArray(result) &&
    result.length > 0 &&
    Array.isArray(result[0]) &&
    result[0].length > 0
  ) {
    const sentimentData = result[0][0];

    if (sentimentData && typeof sentimentData === "object") {
      label =
        typeof sentimentData.label === "string"
          ? sentimentData.label.toUpperCase()
          : "NEUTRAL";
      score =
        typeof sentimentData.score === "number"
          ? sentimentData.score
          : 0.5;

      // Determine sentiment bucket
      if (label === "POSITIVE" && score > 0.5) {
        sentiment = "positive";
      } else if (label === "NEGATIVE" && score > 0.5) {
        sentiment = "negative";
      } else {
        sentiment = "neutral";
      }
    }
  }

  // Update UI
  sentimentResult.classList.add(sentiment);
  sentimentResult.innerHTML = `
        <i class="fas ${getSentimentIcon(sentiment)} icon"></i>
        <span>${label} (${(score * 100).toFixed(1)}% confidence)</span>
    `;

  // 🔥 ВЫЗОВ ФУНКЦИИ ЛОГГИРОВАНИЯ (теперь она доступна!)
  logToGoogleSheet(reviewText.textContent, label, score);
}

// ... остальные функции (getSentimentIcon, showError, hideError) без изменений ...
