# CarValue AI — Web Frontend

A responsive single-page HTML/CSS/JavaScript frontend for the CarValue AI portfolio project.

## Run locally
Open `index.html` in a browser. No build step is needed.

## Important model note
The frontend calls the FastAPI backend configured in `index.html` for Random Forest predictions. The model uses historical listings mainly from 2019–2021, so estimates are not live market quotes.

## DSA features
- **Linked list:** stores successful valuations as linked nodes for a session timeline.
- **Queue:** processes prediction requests FIFO (first in, first out).
- **Stack:** stores successful valuations LIFO (last in, first out) and supports undoing the latest displayed result.

These DSA records are kept in browser memory for the current session and reset on refresh. The ML model and API remain separate.

## Deploy to Netlify from GitHub
1. Push this folder's files to a GitHub repository.
2. In Netlify, choose **Add new project → Import an existing project**, connect GitHub, and select the repository.
3. Set **Build command** to blank and **Publish directory** to `.` (the repository root).
4. Deploy. Future pushes to the production branch will trigger new deploys.
