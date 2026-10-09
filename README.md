# CarValue AI — Web Frontend

A responsive single-page HTML/CSS/JavaScript frontend for the CarValue AI portfolio project.

## Run locally
Open `index.html` in a browser. No build step is needed.

## Important model note
This is a static frontend demo. Its browser-side estimate uses a simplified illustrative formula; it does **not** run the Python Random Forest model and does not query live listings. To serve real ML predictions, deploy the Python model as an API on a Python-capable host and connect this page to that API.

## Deploy to Netlify from GitHub
1. Push this folder's files to a GitHub repository.
2. In Netlify, choose **Add new project → Import an existing project**, connect GitHub, and select the repository.
3. Set **Build command** to blank and **Publish directory** to `.` (the repository root).
4. Deploy. Future pushes to the production branch will trigger new deploys.
