#!/bin/bash

echo "======================================"
echo "Neural Networks Full-Stack Application"
echo "COMP-258 Group Project"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python3 is not installed${NC}"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js is not installed${NC}"
    exit 1
fi

echo -e "${BLUE}[1/5] Setting up Backend...${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "backend/venv" ]; then
    echo "Creating Python virtual environment..."
    cd backend
    python3 -m venv venv
    cd ..
fi

# Activate virtual environment and install dependencies
echo "Installing backend dependencies..."
cd backend
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
pip install -r requirements.txt --quiet
cd ..

echo -e "${GREEN}✓ Backend setup complete${NC}"
echo ""

echo -e "${BLUE}[2/5] Setting up Frontend...${NC}"

# Install frontend dependencies
if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd frontend
    npm install --silent
    cd ..
else
    echo "Frontend dependencies already installed"
fi

echo -e "${GREEN}✓ Frontend setup complete${NC}"
echo ""

echo -e "${BLUE}[3/5] Checking model directory...${NC}"

# Create model directory if it doesn't exist
if [ ! -d "model" ]; then
    echo "Creating model directory..."
    mkdir -p model
    echo "Note: Please ensure your trained models are placed in the 'model' directory"
fi

echo -e "${GREEN}✓ Model directory ready${NC}"
echo ""

echo -e "${BLUE}[4/5] Starting Backend Server...${NC}"

# Start backend in background
cd backend
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
python app.py &
BACKEND_PID=$!
cd ..

echo -e "${GREEN}✓ Backend server started (PID: $BACKEND_PID)${NC}"
echo "   Backend running at: http://localhost:5001"
echo ""

# Wait for backend to start
echo "Waiting for backend to initialize..."
sleep 3

echo -e "${BLUE}[5/5] Starting Frontend Server...${NC}"

# Start frontend
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

echo -e "${GREEN}✓ Frontend server started (PID: $FRONTEND_PID)${NC}"
echo "   Frontend running at: http://localhost:3000"
echo ""

echo "======================================"
echo -e "${GREEN}Application is now running!${NC}"
echo "======================================"
echo ""
echo "Access the application at: http://localhost:3000"
echo "API documentation at: http://localhost:5001/api/health"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "Servers stopped"
    exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup INT

# Wait for user to stop
wait