#!/bin/bash

# Deployment Script to ensure all data and code are synced to production

echo "🚀 Starting deployment process..."

# 1. Add crucial data files (Tournaments, Standings, Fixtures)
echo "📂 Staging data files..."
git add data/tournaments.csv
git add data/standings_ATP.csv
git add data/standings_WTA.csv
git add data/fixtures/*.csv

# 2. Add source code files
echo "💻 Staging code files..."
git add *.py
git add scripts/*.py
git add .streamlit/*
git add requirements.txt

# 3. Check status
echo "📊 Current Status:"
git status

# 4. Prompt for commit message
read -p "📝 Enter commit message: " commit_msg

if [ -z "$commit_msg" ]; then
    echo "❌ Commit message cannot be empty. Aborting."
    exit 1
fi

# 5. Commit and Push
echo "COMMITING: $commit_msg"
git commit -m "$commit_msg"

echo "⬆️ Pushing to origin main..."
git push origin main

echo "✅ Deployment complete! Streamlit Cloud will update shortly."
