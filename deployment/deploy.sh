#!/bin/bash
# Production deployment script for TechGear Electronics Chatbot
# Supports Docker, Docker Compose, and cloud deployments

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

PROJECT_NAME="techgear-chatbot"
IMAGE_NAME="techgear-chatbot-api"
VERSION=${VERSION:-"latest"}
ENVIRONMENT=${ENVIRONMENT:-"production"}

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Pre-deployment Checks
# =============================================================================

check_requirements() {
    log_info "Checking requirements..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if .env file exists
    if [ ! -f .env ]; then
        log_warning ".env file not found. Please create one from .env.production template."
        log_info "Copying .env.production to .env..."
        cp .env.production .env
        log_warning "Please update .env with your actual values before proceeding."
        exit 1
    fi
    
    log_success "All requirements satisfied"
}

# =============================================================================
# Build Functions
# =============================================================================

build_image() {
    log_info "Building Docker image..."
    
    docker build \
        -t "${IMAGE_NAME}:${VERSION}" \
        -t "${IMAGE_NAME}:latest" \
        --build-arg ENVIRONMENT="${ENVIRONMENT}" \
        .
    
    log_success "Docker image built successfully"
}

# =============================================================================
# Deployment Functions
# =============================================================================

deploy_docker() {
    log_info "Deploying with Docker..."
    
    # Stop existing container
    docker stop ${PROJECT_NAME} 2>/dev/null || true
    docker rm ${PROJECT_NAME} 2>/dev/null || true
    
    # Run new container
    docker run -d \
        --name ${PROJECT_NAME} \
        --env-file .env \
        -p 8000:8000 \
        -v "$(pwd)/chroma_db:/app/chroma_db" \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/data:/app/data" \
        --restart unless-stopped \
        "${IMAGE_NAME}:${VERSION}"
    
    log_success "Container deployed successfully"
}

deploy_compose() {
    log_info "Deploying with Docker Compose..."
    
    # Stop existing services
    docker-compose down
    
    # Start services
    docker-compose up -d --build
    
    log_success "Services deployed successfully"
}

# =============================================================================
# Health Check
# =============================================================================

health_check() {
    log_info "Performing health check..."
    
    # Wait for service to start
    sleep 10
    
    # Check health endpoint
    max_attempts=30
    attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            log_success "Health check passed"
            return 0
        fi
        
        attempt=$((attempt + 1))
        log_info "Waiting for service to be ready... ($attempt/$max_attempts)"
        sleep 2
    done
    
    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# =============================================================================
# Rollback
# =============================================================================

rollback() {
    log_warning "Rolling back deployment..."
    
    # Stop current containers
    docker-compose down
    
    # Start with previous version
    VERSION="previous" docker-compose up -d
    
    log_success "Rollback completed"
}

# =============================================================================
# Main Deployment Flow
# =============================================================================

main() {
    log_info "Starting deployment for ${PROJECT_NAME}"
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Version: ${VERSION}"
    
    # Check requirements
    check_requirements
    
    # Build image
    build_image
    
    # Deploy based on method
    case "${1:-compose}" in
        docker)
            deploy_docker
            ;;
        compose)
            deploy_compose
            ;;
        *)
            log_error "Unknown deployment method: $1"
            log_info "Usage: $0 [docker|compose]"
            exit 1
            ;;
    esac
    
    # Health check
    if health_check; then
        log_success "Deployment completed successfully! 🎉"
        log_info "API available at: http://localhost:8000"
        log_info "API docs at: http://localhost:8000/docs"
        log_info "Grafana dashboard at: http://localhost:3000"
        log_info "Prometheus metrics at: http://localhost:9090"
    else
        log_error "Deployment failed health check"
        log_warning "Consider rolling back with: $0 rollback"
        exit 1
    fi
}

# =============================================================================
# Script Entry Point
# =============================================================================

case "${1}" in
    rollback)
        rollback
        ;;
    *)
        main "$@"
        ;;
esac
