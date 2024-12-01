@echo off

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Docker is not running. Please start Docker and try again.
    exit /b 1
)

REM Check if Docker Compose is available
docker compose version >nul 2>&1
if errorlevel 1 (
    echo Docker Compose is not installed. Please install Docker Compose and try again.
    exit /b 1
)

REM Navigate to the project directory
cd /d %~dp0

REM Build the Docker image
docker build -t ffd_gnn_app .

REM Run the Docker container
docker compose up -d

@echo on
