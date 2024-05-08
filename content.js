// Define the scrapeContent function
function scrapeContent() {
  let paragraphs = document.querySelectorAll('textarea, p, h1, h2, h3'); // Select text elements
  let images = document.querySelectorAll('img'); // Select image elements
  let text = '';
  let nonEmptyUrls = [];

  // Scrape text content
  paragraphs.forEach(paragraph => {
    text += paragraph.textContent + '\n';
  });

  // Filter out empty image URLs
  images.forEach(img => {
    let src = img.src;
    if (src.trim() !== '') {
      nonEmptyUrls.push(src);
    }
  });

  // Send scraped text to the background script
  chrome.runtime.sendMessage({ action: "textScraped", text: text });

  // Send non-empty image URLs to the background script
  nonEmptyUrls.forEach(url => {
    chrome.runtime.sendMessage({ action: "imageScraped", image: url });
  });
}

// Listen for page refresh and execute scrapeContent again
window.addEventListener('load', function() {
  scrapeContent();
});
