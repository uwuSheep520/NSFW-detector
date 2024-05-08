// Listen for messages from the content script
chrome.runtime.onMessage.addListener(function(message, sender, sendResponse) {
  if (message.action === "textScraped") {
    console.log("Scraped text:", message.text);
    // Call the function to send the scraped text to the Flask server
    sendTextToFlaskServer(message.text);
  } else if (message.action === "imageScraped") {
    console.log("Scraped image:", message.image);
    // Call the function to send the scraped image URL to the Flask server
    sendImageToFlaskServer(message.image);
  }
});

// Define a function to send the scraped text to the Flask server
function sendTextToFlaskServer(text) {
  fetch('http://localhost:5000/detect_nsfw', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({sentence: text }), // Send the scraped text to the Flask server
  })
  .then(response => {
    if (response.ok) {
      console.log("Connect success.");
      return response.json();
    } else {
      console.log("Connect fail.");
      throw new Error('Network response was not ok.');
    }
  })
  .then(data => {
    // Check if NSFW content is not detected
    if (data.nsfw_not_detected) {
      // Take action when NSFW content is not detected
      console.log("NSFW content detected for text.");
      chrome.tabs.update({ url: "strong.html" });
    } else {
      // NSFW content detected, update the URL to "strong.html"
      console.log("No NSFW content detected for text.");
    }
  })
  .catch(error => {
    console.error('Error:', error);
  });
}
let nsfwPhotoCount = 0;
// Define a function to send the scraped image URL to the Flask server
function sendImageToFlaskServer(imageUrl) {
  fetch('http://localhost:5000/detect_nsfw', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ image: imageUrl }),
  })
    .then(response => {
      if (response.ok) {
        console.log("Connect success.");
        return response.json();
      } else {
        console.log("Connect fail.");
        throw new Error('Network response was not ok.');
      }
    })
    .then(data => {
      if (data && data.result !== undefined) {
        // Check if NSFW content is detected
        if (data.result === 1) {
          console.log("NSFW content detected for image:", imageUrl);
          chrome.tabs.update({ url: "strong.html" });
        } else {
        //pass
        }
      } else {
        console.log("Invalid response format or missing 'result' property.");
      }
    })
    .catch(error => {
      console.error('Error:', error);
    });
  
}
chrome.tabs.onUpdated.addListener(function(tabId, changeInfo, tab) {
  if (changeInfo.status === 'loading') {
    // Reset nsfwPhotoCount to 0 when entering a new page
    nsfwPhotoCount = 0;
  }
});